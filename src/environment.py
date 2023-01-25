import sys
import jax
import time
import jraph
import pickle
from jax import vmap
import jax.numpy as jnp
from functools import partial
from jax import jit

from ops import *


class DockingEnv():
    def __init__(self, dataset, enum, pad, PNRG):
        self.key = PNRG
        self.size = pad
        self.enum = enum
        self.list = list(dataset.keys())
        self.number = len(self.list)
        self.init_rt = {pdb:dataset[pdb]['init_rt'] for pdb in self.list}

        def _extract_data(idx):
            CA = jnp.stack([dataset[pdb]['coord'][idx] for pdb in self.list])
            nodes = jnp.stack([dataset[pdb]['nodes'][idx] for pdb in self.list])
            mask = jnp.stack([dataset[pdb]['masks'][idx] for pdb in self.list])
            if len(CA.shape) == 2: 
                CA = jnp.expand_dims(CA, axis=0)
                mask = jnp.expand_dims(mask, axis=0)
                nodes = jnp.expand_dims(nodes, axis=0)
            return mask, nodes, CA
        
        self.mask_rec, self.feat_rec, self.coord_rec = _extract_data(0)
        self.mask_lig, self.feat_lig, self.coord_lig = _extract_data(1)

        self.padmasks_rec = jnp.where(self.m_rec!=0, 1, 0)
        self.padmasks_lig = jnp.where(self.m_lig!=0, 1, 0)
        self.surfmasks_rec = jnp.where(self.m_rec>=0.2, 0, 1e6)
        self.surfmasks_lig = jnp.where(self.m_lig>=0.2, 0, 1e6)
        self.lengths_rec = jnp.sum(self.padmask_rec, axis=1)
        self.lengths_lig = jnp.sum(self.padmask_lig, axis=1)

        self.edge_rec, self.send_rec, self.neigh_rec = \
                vmap(partial(self.get_edges, self.edge_num, 0))(
                        self.coord_rec, self.coord_rec, self.padmasks_rec, self.padmasks_rec)

        self.edge_lig, self.send_lig, self.neigh_lig = \
                vmap(partial(self.get_edges, self.edge_num, 0))(
                        self.coord_lig, self.coord_lig, self.padmasks_lig, self.padmasks_lig)

        self.ground_truth = vmap(distances_from_coords)(
                self.coord_rec[:,:,None,:], self.coord_lig[:,None,:,:])
        
        self.ground_truth = vmap(lambda x, y, z: x*y[:,None]*z[None,:])(
                self.dmaps, self.padmasks_rec, self.padmasks_lig)


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

#        self.dmaps = vmap(distances_from_coords)(
#                self.coord_rec[:,:,None,:], self.coord_lig[:,None,:,:])
#        
#        self.dmaps = vmap(lambda x, y, z: x*y[:,None]*z[None,:])(
#                self.dmaps, self.padmask_rec, self.padmask_lig)


    def get_state(self):

        edge_int_rec, send_int_rec, neigh_int_rec = \
                vmap(partial(self.get_edges, self.edge_num, 0))(
                        self.coord_rec, self.coord_lig, self.padmasks_rec, self.padmasks_lig)

        edge_int_lig, send_int_lig, neigh_int_lig = \
                vmap(partial(self.get_edges, self.edge_num, 0))(
                        self.coord_lig, self.coord_rec, self.padmasks_lig, self.padmasks_rec)

        edge_int = vmap(lambda x, y: jnp.concatenate((x,y), axis=0))(
                edge_int_rec, edge_int_lig)

        send_int = vmap(lambda x, y: jnp.concatenate((x,y), axis=0))(
                send_int_rec, send_int_lig+self.length_rec)

        neigh_int = vmap(lambda x, y: jnp.concatenate((x,y), axis=0))(
                neigh_int_rec+self.length_rec, neigh_int_lig)

        return edge_int, send_int, neigh_int


    @partial(jax.jit, static_argnums=(0,1))
    def get_edges(self, edge_num, c_i, c_j, mask_i, mask_j):
        '''
        select closest edge_num neighbours to coords in set ca_i 
        between coords in set ca_j, excluding neighbours closer 
        than mindist, and define edge features and indexes using 
        coords information in n_i, n_j, c_i, c_j
        '''

        def _get_residue_edges(edge_num):
      
            def _next_lowest(val, array):
                array = jnp.where(array>val, array, array+10e6)
                next_val = jnp.min(array)
                return next_val, jnp.argwhere(array==next_val, size=1)
      
            sorting_input = jnp.broadcast_to(line[None,:], (edge_num, line.shape[0]))
            sorted_idxs = jnp.squeeze(jax.lax.scan(_next_lowest, 0, sorting_input)[1])
            sorted_line = line[sorted_idxs]
            return sorted_line, sorted_idxs
      
        def _get_distance_features(dist):
            scales = 1.5 ** (jnp.indices((15,))[0]+1)
            return jnp.exp(-((dist ** 2) / scales))

        # find edges indexes
        dmap = distances_from_coords(c_i[:,None,:], c_j[None,:,:])
        dmap = dmap * mask_i[:,None] * mask_j[None,:]

        nodes = jnp.indices((dmap.shape[0],))[0]
        nodes = jnp.broadcast_to(nodes[:,None], (dmap.shape[0], edge_num))
        dist, neighs = vmap(partial(_get_residue_edges, edge_num)(dmap)
        dist, nodes, neighs = jnp.ravel(dist), jnp.ravel(nodes), jnp.ravel(neighs),
        edges = vmap(_get_distance_features)(dist)

        nodes = jnp.array(jnp.where(nodes<true_len, nodes, 799), dtype=jnp.uint16)
        neighs = jnp.array(jnp.where(neighs<true_len, neighs, 799), dtype=jnp.uint16)
        return edges, nodes, neighs


    @partial(jit, static_argnums=(0,))
    def step(self, c_rec, c_lig, actions):
        '''
        apply step rotating full coordinates and recompute distance map
        '''

        def _compute_keypoints(action, c_rec, old_keypts):
            
            def _get_distances(c_rec, min_dist, max_dist, line):
                d1, d2, d3 = line[4:]
                p1, p2, p3 = c_rec[line[1:4]]
                d12 = jnp.linalg.norm(p1-p2)
                max_dist = jnp.max(max_dist, d12)

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

            def _trilateration(c_rec, old_keypt, line):
                p1, p2, p3 = c_rec[line[1:4]]
                r1, r2, r3 = line[4:]

                e_x=(p2-p1)/np.linalg.norm(p2-p1)
                i=np.dot(e_x,(p3-p1))
                e_y=(p3-p1-(i*e_x))/(np.linalg.norm(p3-p1-(i*e_x)))
                e_z=np.cross(e_x,e_y)
                d=np.linalg.norm(p2-p1)
                j=np.dot(e_y,(p3-p1))
                x=((r1**2)-(r2**2)+(d**2))/(2*d)
                y=(((r1**2)-(r3**2)+(i**2)+(j**2))/(2*j))-((i/j)*(x))
                z1=np.sqrt(r1**2-x**2-y**2)
                z2=np.sqrt(r1**2-x**2-y**2)*(-1)
                l1=p1+(x*e_x)+(y*e_y)+(z1*e_z)
                l2=p1+(x*e_x)+(y*e_y)+(z2*e_z)
                
                ### CRIT POINT!!!
                d1 = jnp.linalg.norm(l1-old_keypt)
                d2 = jnp.linalg.norm(l2-old_keypt)
                select = jnp.argmin(jnp.array((d1, d2)))
                return jnp.array((l1, l2))[select]

            distances = vmap(partial(_get_distances, c_rec, 6, 20))(action)
            action = jnp.concatenate((action[:,:4], distances), axis=-1)
            trilateration = partial(_trilateration, c_rec, old_keypts)
            return vmap(trilateration)(action)
        
        
        def _kabsch(new, old):
            new_mean = jnp.mean(new, dim=0, keepdim=True)
            old_mean = jnp.mean(old, dim=0, keepdim=True)

            A = jnp.transpose(new - new_mean) @ (old - old_mean)
            U, S, Vt = jnp.linalg.svd(A)

            Vti = Vt[2, :] * -1
            corr_mat = jnp.diag(jnp.array([1,1,jnp.sign(jnp.linalg.det(A))]))
            midstep = U @ corr_mat
            rot = jnp.array((midstep @ Vt, midstep @ Vti))
            select = jnp.array((jnp.linalg.det(R1), jnp.linalg.det(R2)))

            R = rot[jnp.argmax(select)]
            t = new_mean - jnp.transpose(R @ jnp.transpose(old_mean))
            return R, t


        def _single_step(cloud, rotation, translation, mask):
            
            ### CRIT POINT!!!
            transformed_cloud = (rotation @ cloud.T).T
            transformed_cloud += translation[:,None]
            return transformed_cloud*mask[:,None]


#        def _ge(dmap):
#            edges12, senders12, receivers12,\
#            edges21, senders21, receivers21 = get_interface_edges(dmap, self.enum, self.size)
#
#            edges12 = encode_distances(edges12)
#            edges21 = encode_distances(edges21)
#
#            all_edges = jnp.concatenate((edges12, edges21), axis=0)
#            all_senders = jnp.concatenate((senders12, senders21), axis=0)
#            all_receivers = jnp.concatenate((receivers12, receivers21), axis=0)
#            return all_edges, \
#                   jnp.array(all_senders, dtype=jnp.uint16), \
#                   jnp.array(all_receivers, dtype=jnp.uint16)


        old_keypts = vmap(lambda x, y: x[jnp.flatten(y[:,0])])(c_lig, actions)
        new_keypts = vmap(_compute_keypoints)(action, c_rec, old_keypts)

        Rs, ts = vmap(kabsch)(new_keypts, old_keypts)

        c_lig_new = vmap(_single_step)(c_lig, Rs, ts, self.padmask_lig)

        dmaps = vmap(distances_from_coords)(
                self.c_rec[:,:,None,:], c_lig_new[:,None,:,:])
        
        dmaps = vmap(lambda x, y, z: x*y[:,None]*z[None,:])(
                dmaps_new, self.padmask_rec, self.padmask_lig)
        
        return dmaps, c_lig_new

    @partial(jit, static_argnums=(0,))
    def get_rewards(self, dmaps):
        def _get_reward(dmap, sm_rec, sm_lig):
            contacts = jnp.where((dmap>=6) & (dmap<=8), 1, 0)
            contacts = contacts*sm_rec[:,None]*sm_lig[None,:]
            contacts_rec = jnp.sum(jnp.max(contacts, axis=0))
            contacts_lig = jnp.sum(jnp.max(contacts, axis=1))
            contacts = contacts_rec+contacts_lig

            clashes = jnp.where((dmap<6) & (dmap!=0), (6-dmap)/6, 0)
            clashes = jnp.sum(clashes)
            
            dmap = jnp.where(dmap==0, 1e9, dmap)
            
            return contacts-(clashes+jnp.min(dmap))

        return vmap(_get_reward)(dmaps, self.surf_rec, self.surf_lig)
         

if __name__ == '__main__':
    sys.exit()
