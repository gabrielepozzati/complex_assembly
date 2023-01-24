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
            N = jnp.stack([dataset[pdb]['coord_N'][idx] for pdb in self.list])
            C = jnp.stack([dataset[pdb]['coord_C'][idx] for pdb in self.list])
            CA = jnp.stack([dataset[pdb]['coord_CA'][idx] for pdb in self.list])
            nodes = jnp.stack([dataset[pdb]['nodes'][idx] for pdb in self.list])
            mask = jnp.stack([dataset[pdb]['masks'][idx] for pdb in self.list])
            if len(N.shape) == 2: 
                N = jnp.expand_dims(N, axis=0)
                C = jnp.expand_dims(C, axis=0)
                CA = jnp.expand_dims(CA, axis=0)
                mask = jnp.expand_dims(mask, axis=0)
                nodes = jnp.expand_dims(nodes, axis=0)
            return mask, nodes, N, C, CA
        
        (self.m_rec, self.f_rec,
         self.c_rec_N, self.c_rec_C,
         self.c_rec_CA) = _extract_data(0)

        (self.m_lig, self.f_lig,
         self.c_lig_N, self.c_lig_C, 
         self.c_lig_CA) = _extract_data(1)

        (self.edges_rec, self.nodes_rec, 
         self.neighs_rec) = vmap(partial(self.get_edges, self.edge_num, 0))(
                self.c_rec_N, self.c_rec_C, self.c_rec_CA,
                self.c_lig_N, self.c_lig_C, self.c_lig_CA)

        (self.edges_lig, self.nodes_lig,
         self.neighs_lig) = vmap(partial(self.get_edges, self.edge_num, 0))(
                self.c_lig_N, self.c_lig_C, self.c_lig_CA,
                self.c_rec_N, self.c_rec_C, self.c_rec_CA)

        self.padmasks_rec = jnp.where(self.m_rec!=0, 1, 0)
        self.padmasks_lig = jnp.where(self.m_lig!=0, 1, 0)
        self.surfmasks_rec = jnp.where(self.m_rec>=0.2, 0, 1e6)
        self.surfmasks_lig = jnp.where(self.m_lig>=0.2, 0, 1e6)
        self.lengths_rec = jnp.sum(self.padmask_rec, axis=1)
        self.lengths_lig = jnp.sum(self.padmask_lig, axis=1)

        self.dmaps = vmap(distances_from_coords)(
                self.c_rec_CA[:,:,None,:], self.c_lig_CA[:,None,:,:])
        
        self.dmaps = vmap(lambda x, y, z: x*y[:,None]*z[None,:])(
                self.dmaps, self.padmasks_rec, self.padmasks_lig)


    def reset(self, dataset, idxs):
        '''
        Reset cloud informations for pairs whose index is in idxs. 
        Cloud states are set back to what is stored in dataset.
        '''

        n_clouds, c_clouds, ca_clouds = [], [], []
        for idx, code in enumerate(self.list):
            if idx in idxs: n_clouds.append(dataset[code]['coord_N'][1])
            else: n_clouds.append(self.c_lig_N[idx])
            if idx in idxs: c_clouds.append(dataset[code]['coord_C'][1])
            else: c_clouds.append(self.c_lig_C[idx])
            if idx in idxs: ca_clouds.append(dataset[code]['coord_CA'][1])
            else: ca_clouds.append(self.c_lig_CA[idx])
        self.c_lig_N = jnp.stack(n_clouds)
        self.c_lig_C = jnp.stack(c_clouds)
        self.c_lig_CA = jnp.stack(ca_clouds)

        (self.edges_rec, self.nodes_rec,
         self.neighs_rec) = vmap(partial(self.get_edges, self.edge_num, 0))(
                self.c_rec_N, self.c_rec_C, self.c_rec_CA,
                self.c_lig_N, self.c_lig_C, self.c_lig_CA)

        (self.edges_lig, self.nodes_lig,
         self.neighs_lig) = vmap(partial(self.get_edges, self.edge_num, 0))(
                self.c_lig_N, self.c_lig_C, self.c_lig_CA,
                self.c_rec_N, self.c_rec_C, self.c_rec_CA)

        self.dmaps = vmap(distances_from_coords)(
                self.c_rec_CA[:,:,None,:], self.c_lig_CA[:,None,:,:])
        
        self.dmaps = vmap(lambda x, y, z: x*y[:,None]*z[None,:])(
                dmaps, self.padmask_rec, self.padmask_lig)


    @partial(jax.jit, static_argnums=(0,1,2))
    def get_edges(self, edge_num, min_dist, ca_i, ca_j):
        '''
        select closest edge_num neighbours to coords in set ca_i 
        between coords in set ca_j, excluding neighbours closer 
        than mindist, and define edge features and indexes using 
        coords information in n_i, n_j, c_i, c_j
        '''

        def _get_residue_edges(edge_num, min_dist, line):
      
            def _next_lowest(min_val, array):
                array = jnp.where(array>min_val, array, array+10e6)
                next_val = jnp.min(array)
                return next_val, jnp.argwhere(array==next_val, size=1)
      
            sorting_input = jnp.broadcast_to(line[None,:], (edge_num, line.shape[0]))
            sorted_idxs = jnp.squeeze(jax.lax.scan(_next_lowest, min_dist, sorting_input)[1])
            sorted_line = line[sorted_idxs]
            return sorted_line, sorted_idxs
      
#        def _get_local_frames(n, c, ca):
#            u_i = (n-ca)/jnp.linalg.norm((n-ca))
#            t_i = (c-ca)/jnp.linalg.norm((c-ca))
#            n_i = jnp.cross(u_i, t_i)/jnp.linalg.norm(jnp.cross(u_i, t_i))
#            v_i = jnp.cross(n_i, u_i)
#            return jnp.stack((n_i, u_i, v_i), axis=0)

#        def _get_relative_features(local_i, local_j, ca_i, ca_j):
#            q_ij = jnp.matmul(local_i, (ca_j-ca_i))
#            k_ij = jnp.matmul(local_i, local_j[0])
#            t_ij = jnp.matmul(local_i, local_j[1])
#            s_ij = jnp.matmul(local_i, local_j[2])
#            feats = jnp.concatenate((q_ij, k_ij, t_ij, s_ij), axis=0)
#            return jnp.squeeze(feats)
      
        def _get_distance_features(dist):
            scales = 1.5 ** (jnp.indices((15,))[0]+1)
            return jnp.exp(-((dist ** 2) / scales))
      
        # find edges indexes
        dmap = distances_from_coords(ca_i[:,None,:], ca_j[None,:,:])
        nodes = jnp.indices((dmap.shape[0],))[0]
        nodes = jnp.broadcast_to(nodes[:,None], (dmap.shape[0], edge_num))
        dist, neighs = vmap(partial(_get_residue_edges, edge_num, min_dist)(dmap)
        dist, nodes, neighs = jnp.ravel(dist), jnp.ravel(nodes), jnp.ravel(neighs),
        
#        # compute residues local frames
#        m_get_local_frames = vmap(_get_local_frames)
#        all_local_i = m_get_local_frames(n_i, c_i, ca_i)
#        all_local_j = m_get_local_frames(n_j, c_j, ca_j)

#        # select edge-specific local frames
#        local_i, cloud_i = all_local_i[nodes], ca_i[nodes]
#        local_j, cloud_j = all_local_j[neighs], ca_j[neighs]
        
#        # get edge features
#        rfeats = vmap(_get_relative_features)(local_i, local_j, cloud_i, cloud_j)
        edges = vmap(_get_distance_features)(dist)
#        edges = jnp.concatenate((rfeats, dfeats), axis=-1)
      
        return edges, nodes, neighs


    @partial(jit, static_argnums=(0,))
    def step(self, c_rec_CA, c_lig_CA, actions):
        '''
        apply step rotating full coordinates and recompute distance map
        '''

        def _compute_keypoints(action, c_rec_CA, old_keypts):
            
            def _get_distances(c_rec_CA, min_dist, max_dist, line):
                d1, d2, d3 = line[4:]
                p1, p2, p3 = c_rec_CA[line[1:4]]
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

            def _trilateration(c_rec_CA, old_keypt, line):
                p1, p2, p3 = line[1:4]
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

            distances = vmap(partial(_get_distances, c_rec_CA, min_dist, max_dist))(action)
            action = jnp.concatenate((action[:,:4], distances), axis=-1)
            trilateration = partial(_trilateration, c_rec_CA, old_keypts)
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


        def _get_interface_graph(dmap):
            edges12, senders12, receivers12,\
            edges21, senders21, receivers21 = get_interface_edges(dmap, self.enum, self.size)

            edges12 = encode_distances(edges12)
            edges21 = encode_distances(edges21)

            all_edges = jnp.concatenate((edges12, edges21), axis=0)
            all_senders = jnp.concatenate((senders12, senders21), axis=0)
            all_receivers = jnp.concatenate((receivers12, receivers21), axis=0)
            return all_edges, \
                   jnp.array(all_senders, dtype=jnp.uint16), \
                   jnp.array(all_receivers, dtype=jnp.uint16)


        old_keypts = vmap(lambda x, y: x[jnp.flatten(y[:,0])])(c_lig_CA, actions)
        new_keypts = vmap(_compute_keypoints)(action, c_rec_CA, old_keypts)

        Rs, ts = vmap(kabsch)(new_keypts, old_keypts)

        c_lig_CA = vmap(_single_step)(c_lig_CA, Rs, ts, self.padmask_lig)

        dmaps = vmap(distances_from_coords)(
                self.c_rec_CA[:,:,None,:], c_lig_CA_new[:,None,:,:])
        
        dmaps = vmap(lambda x, y, z: x*y[:,None]*z[None,:])(
                dmaps_new, self.padmask_rec, self.padmask_lig)
        
        return dmaps, c_lig_N, c_lig_C, c_lig_CA

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
