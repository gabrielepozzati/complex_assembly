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
from graph import *


class DockingEnv():
    def __init__(self, dataset, enum, pad, PNRG):
        self.key = PNRG
        self.size = pad
        self.enum = enum
        self.list = list(dataset.keys())
        self.number = len(self.list)
        self.init_rt = {pdb:dataset[pdb]['init_rt'] for pdb in self.list}
        
        def _extract_data(idx):
            N = jnp.stack([dataset[pdb]['coord_N'][idx] for pdb in self.list]}
            C = jnp.stack([dataset[pdb]['coord_C'][idx] for pdb in self.list])
            CA = jnp.stack([dataset[pdb]['coord_CA'][idx] for pdb in self.list])
            mask = jnp.stack([dataset[pdb]['masks'][idx] for pdb in self.list])
            nodes = jnp.stack([dataset[pdb]['nodes'][idx] for pdb in self.list])
            return mask, nodes, N, C, CA
        
        (self.m_rec, 
         self.n_rec,
         self.c_rec_N,
         self.c_rec_C,
         self.c_rec_CA) = _extract_data(0)

        (self.m_lig,
         self.n_lig,
         self.c_lig_N,
         self.c_lig_C, 
         self.c_lig_CA) = _extract_data(1)

        self.padmask_rec = jnp.where(self.m_rec!=0, 1, 0)
        self.padmask_lig = jnp.where(self.m_lig!=0, 1, 0)
        self.surfmask_rec = jnp.where(self.m_rec>=0.2, 0, 1e6)
        self.surfmask_lig = jnp.where(self.m_lig>=0.2, 0, 1e6)
        self.l_rec = jnp.sum(self.padmask_rec, axis=1)
        self.l_lig = jnp.sum(self.padmask_lig, axis=1)

        self.distances_from_coords = vmap(distances_from_coords)
        
        self.dmaps = self.distances_from_coords(
                self.c_rec_CA[:,:,None,:], self.c_lig_CA[:,None,:,:])
        
        self.dmaps = vmap(lambda x, y, z: x*y[:,None]*z[None,:])(
                dmaps, self.padmask_rec, self.padmask_lig)


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

        self.dmaps = self.distances_from_cloud(
                self.c_rec_CA[:,:,None,:], self.c_lig_CA[:,None,:,:])
        
        self.dmaps = vmap(lambda x, y, z: x*y[:,None]*z[None,:])(
                dmaps, self.padmask_rec, self.padmask_lig)


#    def get_partial_feats(self, dmaps, thr):
#        '''
#        Select subset of nodes for each chain in each pair. Nodes has to be
#        solvent accessible (RSASA>0.2) and close to interface (distance from
#        paired chain < thr). Find the longest set of nodes, round it to the 
#        next largest multiple of 50 and pad every pair to this length.
#        '''

    @jax.jit
    def get_surface_distances(dmaps):
        # massively increase value of padding and buried residues distances
        dmaps = vmap(lambda x, y, z: x+y[:,None]+z[None,:])(
                dmaps, self.surfmask_rec, self.surfmask_lig)
        # find closest residues to the paired chain for each chain
        m_min = vmap(lambda x, y: jnp.min(x, axis=y))
        rec_closest = m_min(dmaps, 0)
        lig_closest = m_min(dmaps, 1)
        # count how many residues are interface/close to interface
        rec_max = jnp.sum(jnp.where(rec_closest<=thr, 1, 0), axis=1)
        lig_max = jnp.sum(jnp.where(lig_closest<=thr, 1, 0), axis=1)
        # find largest interface
        # max_size = jnp.max(jnp.concatenate((rec_max, lig_max), axis=-1))
        return rec_closest, lig_closest#, max_size

#        def _get_and_pad(max_size, array, idxs):
#            array = array[idxs]
#            pad_size = max_size-array.shape[0]
#            return jnp.pad(array, ((0,pad_size),(0,0)))
#
#        # find closest residues in the pair for both chains
#        rec_closest, lig_closest, max_size = get_surface_distances(dmaps)
#        # round in excess to the closest multiple of 50 to find ideal padding
#        while max_size % 50 != 0: max_size += 1
#        # extract and pad coordinates
#        rec_idx = jnp.argwhere(rec_closest<=thr)
#        lig_idx = jnp.argwhere(lig_closest<=thr)
#
#        m_get_and_pad = vmap(partial(_get_and_pad, max_size))
#        
#        c_rec_CA = m_get_and_pad(self.c_rec_CA, rec_idx)
#        c_rec_C = m_get_and_pad(self.c_rec_C, rec_idx)
#        c_rec_N = m_get_and_pad(self.c_rec_N, rec_idx)
#        n_rec = m_get_and_pad(self.n_rec, rec_idx)
#
#        c_lig_CA = m_get_and_pad(self.c_lig_CA, lig_idx)
#        c_lig_C = m_get_and_pad(self.c_lig_C, lig_idx)
#        c_lig_N = m_get_and_pad(self.c_lig_N, lig_idx)
#        n_lig = m_get_and_pad(self.n_lig, lig_idx)
#
#        return c_rec_CA, c_rec_C, c_rec_N, n_rec,
#               c_lig_CA, c_lig_C, c_lig_N, n_lig


    @partial(jax.jit, static_argnums=(0,7))
    def get_edges(self, n_i, n_j, c_i, c_j, ca_i, ca_j, enum):
        '''
        select closest enum neighbours to coords in set ca_i between coords in set ca_j 
        for each node and define edge features and indexes using coords information in
        n_i, n_j, c_i, c_j
        '''

        def _get_residue_edges(line):
      
            def _next_lowest(min_val, array):
                array = jnp.where(array>min_val, array, array+10e6)
                next_val = jnp.min(array)
                return next_val, jnp.argwhere(array==next_val, size=1)
      
            sorting_input = jnp.broadcast_to(line[None,:], (enum, line.shape[0]))
            sorted_idxs = jnp.squeeze(jax.lax.scan(_next_lowest, 0, sorting_input)[1])
            sorted_line = line[sorted_idxs]
            return sorted_line, sorted_idxs
      
        def _get_local_frames(n, c, ca):
            u_i = (n-ca)/jnp.linalg.norm((n-ca))
            t_i = (c-ca)/jnp.linalg.norm((c-ca))
            n_i = jnp.cross(u_i, t_i)/jnp.linalg.norm(jnp.cross(u_i, t_i))
            v_i = jnp.cross(n_i, u_i)
            return jnp.stack((n_i, u_i, v_i), axis=0)

        def _get_relative_features(local_i, local_j, ca_i, ca_j):
            q_ij = jnp.matmul(local_i, (ca_j-ca_i))
            k_ij = jnp.matmul(local_i, local_j[0])
            t_ij = jnp.matmul(local_i, local_j[1])
            s_ij = jnp.matmul(local_i, local_j[2])
            feats = jnp.concatenate((q_ij, k_ij, t_ij, s_ij), axis=0)
            return jnp.squeeze(feats)
      
        def _get_distance_features(dist):
            scales = 1.5 ** (jnp.indices((15,))[0]+1)
            return jnp.exp(-((dist ** 2) / scales))
      
        dmap = self.distances_from_coords(ca_i[:,:,None,:], ca_j[:,:,None,:])
        nodes_i = jnp.indices((dmap.shape[0],))[0]
        nodes_i = jnp.broadcast_to(nodes_i[:,None], (dmap.shape[0], enum))
        dist, neighs_j = vmap(_get_residue_edges)(cmap)
        dist, nodes_i, neighs_j = jnp.ravel(dist), jnp.ravel(nodes_i), jnp.ravel(neighs_j),

        m_get_local_frames = vmap(_get_local_frames)
        local_i = m_get_local_frames(n_i, c_i, ca_i)
        local_j = m_get_local_frames(n_j, c_j, ca_j)

        local_i, cloud_i = local_i[nodes_i], ca_i[nodes_i]
        local_j, cloud_j = local_j[neighs_j], ca_j[neighs_j]
        
        rfeats = vmap(_get_relative_features)(local_i, local_j, cloud_i, cloud_j)
        dfeats = vmap(_get_distance_features)(dist)
        edges = jnp.concatenate((rfeats, dfeats), axis=-1)
      
        return (edges, nodes_i, neighs_j)


    @partial(jit, static_argnums=(0,))
    def step(self, c_lig_N, c_lig_C, c_lig_CA, actions):
        '''
        apply step rotating full coordinates and recompute distance map
        '''

        def _single_step(cloud, action, mask, length):
            geom_center = jnp.sum(cloud, axis=0)/length
            rotation = jnp.reshape(action[:9], shape=(3,3))
            rotate = vmap(partial(lambda x, y: x@y, rotation))

            ligand_frame = cloud - geom_center[None,:]
            ligand_frame = rotate(ligand_frame)
            #transformed_cloud = quat_rotation(ligand_frame, action[:4])
            cloud = ligand_frame + geom_center[None,:]
            return (transformed_cloud+action[None,9:-1])*mask[:,None]

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

        c_lig_N = vmap(_single_step)(c_lig_N, actions, self.padmask_lig, self.l_lig)
        c_lig_C = vmap(_single_step)(c_lig_C, actions, self.padmask_lig, self.l_lig)
        c_lig_CA = vmap(_single_step)(c_lig_CA, actions, self.padmask_lig, self.l_lig)

        dmaps = self.distances_from_cloud(
                self.c_rec_CA[:,:,None,:], c_lig_CA_new[:,None,:,:])
        
        dmaps = vmap(lambda x, y, z: x*y[:,None]*z[None,:])(
                dmaps_new, self.padmask_rec, self.padmask_lig)
        
        return dmaps_new, c_lig_new, e_int, s_int, r_int

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
