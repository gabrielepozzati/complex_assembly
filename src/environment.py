import sys
import jax
import time
import jraph
import pickle
from jax import vmap
import jax.numpy as jnp
import jax.random as jrn
from functools import partial
from jax import jit

from ops import *


class DockingEnv():
    def __init__(self, dataset, enum, pad, PNRG):
        self.key = PNRG
        self.size = pad
        self.list = list(dataset.keys())
        self.number = len(self.list)
        self.edge_num = enum

        def _extract_data(idx):
            CA = jnp.stack([dataset[pdb]['coord'][idx] for pdb in self.list])
            nodes = jnp.stack([dataset[pdb]['nodes'][idx] for pdb in self.list])
            mask = jnp.stack([dataset[pdb]['masks'][idx] for pdb in self.list])
            if len(CA.shape) == 2: 
                CA = jnp.expand_dims(CA, axis=0)
                mask = jnp.expand_dims(mask, axis=0)
                nodes = jnp.expand_dims(nodes, axis=0)
            return mask, nodes, CA
        
        def _get_dmaps(c_i, c_j, m_i, m_j):
            dmaps = vmap(distances_from_coords)(
                    c_i[:,:,None,:], c_j[:,None,:,:])

            return  vmap(lambda x, y, z: x*y[:,None]*z[None,:])(
                    dmaps, m_i, m_j)

        self.mask_rec, self.feat_rec, self.coord_rec = _extract_data(0)
        self.mask_lig, self.feat_lig, self.coord_lig = _extract_data(1)

        self.padmasks_rec = jnp.where(self.mask_rec!=0, 1, 0)
        self.padmasks_lig = jnp.where(self.mask_lig!=0, 1, 0)
        
        self.surfmasks_rec = jnp.where(self.mask_rec>=0.1, 1, 0)
        self.surfmasks_lig = jnp.where(self.mask_lig>=0.1, 1, 0)
        
        self.lengths_rec = jnp.sum(self.padmasks_rec, axis=1)
        self.lengths_lig = jnp.sum(self.padmasks_lig, axis=1)
        
        self.rmaps = _get_dmaps(self.coord_rec, self.coord_rec,
                self.padmasks_rec, self.padmasks_rec)
        self.lmaps = _get_dmaps(self.coord_lig, self.coord_lig,
                self.padmasks_lig, self.padmasks_lig)
        self.true_int = _get_dmaps(self.coord_rec, self.coord_lig,
                self.padmasks_rec, self.padmasks_lig)

        self.edge_rec, self.send_rec, self.neigh_rec = \
                vmap(partial(self.get_edges, self.edge_num))(
                        self.lengths_rec, self.lengths_rec, self.rmaps)
        self.edge_lig, self.send_lig, self.neigh_lig = \
                vmap(partial(self.get_edges, self.edge_num))(
                        self.lengths_lig, self.lengths_lig, self.lmaps)

        self.rmaps = vmap(lambda x,y,z: x*y[:,None]*z[None,:])(
                self.rmaps, self.surfmasks_rec, self.surfmasks_rec)
        self.lmaps = vmap(lambda x,y,z: x*y[:,None]*z[None,:])(
                self.lmaps, self.surfmasks_lig, self.surfmasks_lig)


    def reset(self, dataset, idxs):
        '''
        Reset cloud informations for pairs whose index is in idxs. 
        Cloud states are set back to what is stored in dataset.
        '''

        ca_clouds = []
        for idx, code in enumerate(self.list):
            if idx in idxs: ca_clouds.append(dataset[code]['coord_CA'][1])
            else: ca_clouds.append(self.coord_lig[idx])
        self.coord_lig = jnp.stack(ca_clouds)


    def get_state(self, coord_lig):

        dmaps = vmap(distances_from_coords)(
                self.coord_rec[:,:,None,:], coord_lig[:,None,:,:])

        dmaps = vmap(lambda x, y, z: x*y[:,None]*z[None,:])(
                dmaps, self.padmasks_rec, self.padmasks_lig)

        dmaps = vmap(lambda x, y, z: x*y[:,None]*z[None,:])(
                dmaps, self.surfmasks_rec, self.surfmasks_lig)

        edge_int_rec, send_int_rec, neigh_int_rec = \
                vmap(partial(self.get_edges, self.edge_num))(
                        self.lengths_rec, self.lengths_lig, dmaps)

        dmaps_t = vmap(lambda x: x.T)(dmaps)

        edge_int_lig, send_int_lig, neigh_int_lig = \
                vmap(partial(self.get_edges, self.edge_num))(
                        self.lengths_lig, self.lengths_rec, dmaps_t)

        edge_int = jnp.concatenate((edge_int_rec, edge_int_lig), axis=1)

        send_int = jnp.concatenate(
                (send_int_rec, 
                 send_int_lig+self.lengths_rec[:,None]), axis=1)

        neigh_int = jnp.concatenate(
                (neigh_int_rec+self.lengths_rec[:,None], 
                 neigh_int_lig), axis=1)

        dmaps = jnp.where(dmaps!=0., dmaps, 1e6)
        mindist_rec = vmap(lambda x:jnp.min(x, axis=1))(dmaps)
        mindist_lig = vmap(lambda x:jnp.min(x, axis=0))(dmaps)

        intmask_rec = vmap(lambda x: 
                jnp.where((4<x)&(x<=8), 1., 0.))(mindist_rec)
        intmask_lig = vmap(lambda x: 
                jnp.where((4<x)&(x<=8), 1., 0.))(mindist_lig)

        rimmask_rec = vmap(lambda x: 
                jnp.where((8<x)&(x<=12), 1., 0.))(mindist_rec)
        rimmask_lig = vmap(lambda x: 
                jnp.where((8<x)&(x<=12), 1., 0.))(mindist_lig)

        return (edge_int, send_int, neigh_int, 
                intmask_rec, intmask_lig, rimmask_rec, rimmask_lig)


    @partial(jax.jit, static_argnums=(0,1))
    def get_edges(self, edge_num, l1, l2, dmap):
        '''
        select closest edge_num neighbours to coords in set ca_i 
        between coords in set ca_j, excluding neighbours closer 
        than mindist, and define edge features and indexes using 
        coords information in n_i, n_j, c_i, c_j
        '''

        def _get_residue_edges(edge_num, line):
      
            def _next_lowest(val, line):
                line = jnp.where(line>val, line, line+10e6)
                next_val = jnp.min(line)
                return next_val, jnp.argwhere(line==next_val, size=1)
      
            sorting_input = jnp.broadcast_to(line[None,:], (edge_num, line.shape[0]))
            sorted_idxs = jnp.squeeze(jax.lax.scan(_next_lowest, 0, sorting_input)[1])
            sorted_line = line[sorted_idxs]
            return sorted_line, sorted_idxs
      
        def _get_distance_features(dist):
            scales = 1.5 ** (jnp.indices((15,))[0]+1)
            return jnp.exp(-((dist ** 2) / scales))

        nodes = jnp.indices((dmap.shape[0],))[0]
        nodes = jnp.broadcast_to(nodes[:,None], (dmap.shape[0], edge_num))
        dist, neighs = vmap(partial(_get_residue_edges, edge_num))(dmap)
        dist, nodes, neighs = jnp.ravel(dist), jnp.ravel(nodes), jnp.ravel(neighs),
        edges = vmap(_get_distance_features)(dist)

        nodes = jnp.array(jnp.where(nodes<l1, nodes, 799), dtype=jnp.uint16)
        neighs = jnp.array(jnp.where(neighs<l2, neighs, 799), dtype=jnp.uint16)
        return edges, nodes, neighs


    #@partial(jit, static_argnums=(0,))
    def step(self, c_rec, c_lig, actions):
        '''
        apply step rotating full coordinates and recompute distance map
        '''
        @jax.jit
        def _compute_keypoints(idxs, dist, c_rec, old_keypts):
            
            def _get_distances(c_rec, min_dist, max_dist, idxs, dist):
                d1, d2, d3 = dist
                p1, p2, p3 = c_rec[idxs[1:]]
                d12 = jnp.linalg.norm(p1-p2)
                max_dist = jnp.maximum(max_dist, d12)

                # compute distance between p1 and ligand point
                d1l = (max_dist-min_dist)*d1+min_dist
                
                # compute lower distance limit btw p2 and ligand point
                min_d2l = jnp.abs((d1l-d12))

                # compute distance between p2 and ligand point
                d2l = (max_dist-min_d2l)*d2+min_d2l

                # compute center/radius/normal of intersection circle
                # btw sphere1 and sphere2
                d_p1_cc = 0.5*(d12+(d1l**2-d2l**2)/d12)
                cc = p1+d_p1_cc*(p2-p1)/d12
                cr = jnp.sqrt(d1l**2-d_p1_cc**2)
                cn = (p1-cc)/jnp.linalg.norm(p1-cc)

                # compute lower and upper distance limits btw p3 and ligand point
                delta = p3 - cc
                Ndelta = jnp.dot(cn, delta)
                crossNdelta = jnp.cross(cn,delta)
                radial_low = jnp.linalg.norm(crossNdelta) - cr
                radial_high = jnp.linalg.norm(crossNdelta) + cr
                min_d3l = jnp.sqrt(Ndelta**2+radial_low**2)
                max_d3l = jnp.sqrt(Ndelta**2+radial_high**2)

                # compute distance between p3 and ligand point
                d3l = (max_d3l-min_d3l)*d3+min_d3l

                return jnp.array([d1l, d2l, d3l])


            def _trilateration(c_rec, idxs, dist):
                p1, p2, p3 = c_rec[idxs[1:]]
                r1, r2, r3 = dist

                e_x=(p2-p1)/jnp.linalg.norm(p2-p1)
                i=jnp.dot(e_x,(p3-p1))
                e_y=(p3-p1-(i*e_x))/(jnp.linalg.norm(p3-p1-(i*e_x)))
                e_z=jnp.cross(e_x,e_y)
                d=jnp.linalg.norm(p2-p1)
                j=jnp.dot(e_y,(p3-p1))
                x=((r1**2)-(r2**2)+(d**2))/(2*d)
                y=(((r1**2)-(r3**2)+(i**2)+(j**2))/(2*j))-((i/j)*(x))
                z1 = jnp.sqrt(r1**2-x**2-y**2)
                z2 = jnp.sqrt(r1**2-x**2-y**2)*(-1)
                l1 = p1+(x*e_x)+(y*e_y)+(z1*e_z)
                l2 = p1+(x*e_x)+(y*e_y)+(z2*e_z)
                
                c_rec_mean = jnp.mean(c_rec, axis=0)
                d1 = jnp.linalg.norm(l1-c_rec_mean)
                d2 = jnp.linalg.norm(l2-c_rec_mean)
                select = jnp.argmax(jnp.array((d1,d2)))
                return jnp.array((l1, l2))[select]
            
            dist = vmap(partial(_get_distances, c_rec, 4, 12))(idxs, dist)
            trilateration = partial(_trilateration, c_rec)
            return vmap(trilateration)(idxs, dist)

        
        def _kabsch(new, old):
            new_mean = jnp.mean(new, axis=0, keepdims=True)
            old_mean = jnp.mean(old, axis=0, keepdims=True)

            A = jnp.transpose(new - new_mean) @ (old - old_mean)
            U, S, Vt = jnp.linalg.svd(A)
            V = Vt.T
            U = U.T
           
            corr_mat = jnp.diag(jnp.array([1,1,jnp.sign(jnp.linalg.det(V@U))]))
            R = V @ (corr_mat @ U)
            t = old_mean - jnp.transpose(R @ jnp.transpose(new_mean))
            return R, jnp.squeeze(t)

        @jax.jit
        def _single_step(cloud, rotation, translation, mask):
            
            cloud = (rotation @ cloud.T).T
            cloud += translation[None,:]
            return cloud*mask[:,None]

        dist = jnp.array(actions[:,:,4:])
        idxs = jnp.array(actions[:,:,:4], dtype=jnp.uint16)

        old_keypts = vmap(lambda x, y: x[jnp.ravel(y[:,0])])(c_lig, idxs)
        new_keypts = vmap(_compute_keypoints)(idxs, dist, c_rec, old_keypts) 

        Rs, ts = vmap(_kabsch)(new_keypts, old_keypts)
        c_lig = vmap(_single_step)(c_lig, Rs, ts, self.padmasks_lig)

        return c_lig, Rs, ts

    @partial(jit, static_argnums=(0,))
    def get_rewards(self, dmaps):
        def _get_reward(dmap, sm_rec, sm_lig):
            contacts = jnp.where((6<=dmap) & (dmap<=8), 1, 0)
            contacts = contacts*sm_rec[:,None]*sm_lig[None,:]
            contacts_rec = jnp.sum(jnp.max(contacts, axis=0))
            contacts_lig = jnp.sum(jnp.max(contacts, axis=1))
            contacts = contacts_rec+contacts_lig

            clashes = jnp.where((dmap<6) & (dmap!=0), (6-dmap)/6, 0)
            clashes = jnp.sum(clashes)
            
            dmap = jnp.where(dmap==0, 1e9, dmap)
            
            return contacts, clashes, jnp.min(dmap)

        return vmap(_get_reward)(dmaps, self.surfmasks_rec, self.surfmasks_lig)
         
    def get_random_action(self, key):

        def _compose_action(key, dmap, rmap, lmap, 
                intmask_rec, intmask_lig, rimmask_rec, rimmask_lig):
            
            def _next_lowest(val, line):
                line = jnp.where(line>val, line, line+1e4)
                next_val = jnp.min(line)
                return next_val, jnp.argwhere(line==next_val, size=1)

            dmap_rim = dmap*rimmask_rec[:,None]*rimmask_lig[None,:]
            dmap_rim = jnp.where(dmap_rim!=0, dmap_rim, 1e6)

            # choose l1 in the ligand interface rim
            lig_mind = jnp.min(dmap_rim, axis=0)
            lig_mind = jnp.broadcast_to(lig_mind[None,:], (10,lig_mind.shape[0]))
            lig_rim_idxs = jnp.squeeze(jax.lax.scan(_next_lowest, 0, lig_mind)[1])
            key, keyn = jrn.split(key, 2)
            l1 = jrn.choice(keyn, lig_rim_idxs)
            
            # find l2 as closest to l1 in the interface
            lmap_int = lmap*intmask_lig[None,:]
            lmap_int = jnp.where(lmap_int!=0, lmap_int, 1e6)
            l2 = jnp.argmin(lmap_int[l1])

            # find l3 != from l2, as closest to l1 in the interface
            lmap_int = lmap*intmask_lig[None,:]
            lmap_int = lmap_int.at[:,l2].set(0)
            lmap_int = jnp.where(lmap_int!=0, lmap_int, 1e6)
            l3 = jnp.argmin(lmap_int[l1])

            # choose r11 between the 3 receptor rim residues 
            # closest to residues l1 in ligand rim 
            rec_l1d = dmap*rimmask_rec[:,None]
            rec_l1d = jax.lax.dynamic_slice(rec_l1d, (0,l1), (self.size,1))
            rec_l1d = jnp.squeeze(rec_l1d)
            rec_l1d = jnp.broadcast_to(rec_l1d[None,:], (3,rec_l1d.shape[0]))
            rec_rim_idxs = jnp.squeeze(jax.lax.scan(_next_lowest, 0, rec_l1d)[1])
            key, keyn = jrn.split(key, 2)
            r11 = jrn.choice(keyn, rec_rim_idxs)

            # find r12 and r13 as closest to r11
            rmap_int = rmap*intmask_rec[None,:]
            rmap_int = jnp.where(rmap_int!=0, rmap_int, 1e6)
            r12 = jnp.argmin(rmap_int[r11])

            rmap_int = rmap*intmask_rec[None,:]
            rmap_int = rmap_int.at[:,r12].set(0)
            rmap_int = jnp.where(rmap_int!=0, rmap_int, 1e6)
            r13 = jnp.argmin(rmap_int[r11])

            ir_mask_rec = jnp.where((intmask_rec==1)|(rimmask_rec==1), 1, 0)
            # find r21, r22 and r23 as closest to l2 in interface+rim
            rec_l2d = dmap*ir_mask_rec[:,None]
            rec_l2d = jax.lax.dynamic_slice(rec_l2d, (0,l2), (self.size,1))
            rec_l2d = jnp.squeeze(rec_l2d)
            rec_l2d = jnp.broadcast_to(rec_l2d[None,:], (3,rec_l2d.shape[0]))
            r21, r22, r23 = jnp.squeeze(jax.lax.scan(_next_lowest, 0, rec_l2d)[1])

            # find r31, r32 and r33 as closest to l3 in interface+rim
            rec_l3d = dmap*ir_mask_rec[:,None]
            rec_l3d = jax.lax.dynamic_slice(rec_l3d, (0,l3), (self.size,1))
            rec_l3d = jnp.squeeze(rec_l3d)
            rec_l3d = jnp.broadcast_to(rec_l3d[None,:], (3,rec_l3d.shape[0]))
            r31, r32, r33 = jnp.squeeze(jax.lax.scan(_next_lowest, 0, rec_l3d)[1])

            # compose action indexes
            ligs = jnp.array((l1, l2, l3))
            recs = jnp.stack((
                    jnp.array((r11, r12, r13)), 
                    jnp.array((r21, r22, r23)), 
                    jnp.array((r31, r32, r33))))

            idxs = jnp.concatenate((ligs[:, None], recs), axis=1)

            # get distances fixers and return full action
            key, keyn = jrn.split(key, 2)
            dist = jrn.normal(key, shape=(3,3))*0.05+0.5
            dist = jnp.clip(dist, a_min=0, a_max=1)
            return jnp.concatenate((idxs, dist), axis=1)

        dmaps = vmap(distances_from_coords)(
                self.coord_rec[:,:,None,:], self.coord_lig[:,None,:,:])
        dmaps = vmap(lambda x,y,z: x*y[:,None]*z[None,:])(
                dmaps, self.surfmasks_rec, self.surfmasks_lig)
        dmaps = vmap(lambda x,y,z: x*y[:,None]*z[None,:])(
                dmaps, self.padmasks_rec, self.padmasks_lig)
        
        return  vmap(partial(_compose_action, key))(
                dmaps, self.rmaps, self.lmaps, 
                self.intmask_rec, self.intmask_lig, 
                self.rimmask_rec, self.rimmask_lig)


if __name__ == '__main__':
    sys.exit()
