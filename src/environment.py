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
from quaternions import *

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

        self.mask_recs, self.feat_recs, self.coord_recs = _extract_data(0)
        self.mask_ligs, self.feat_ligs, self.coord_ligs = _extract_data(1)

        self.padmask_recs = jnp.where(self.mask_recs!=0, 1, 0)
        self.padmask_ligs = jnp.where(self.mask_ligs!=0, 1, 0)
        
        self.length_recs = jnp.sum(self.padmask_recs, axis=1)
        self.length_ligs = jnp.sum(self.padmask_ligs, axis=1)
        
        self.rmaps = _get_dmaps(self.coord_recs, self.coord_recs,
                self.padmask_recs, self.padmask_recs)
        self.lmaps = _get_dmaps(self.coord_ligs, self.coord_ligs,
                self.padmask_ligs, self.padmask_ligs)
        self.true_ints = _get_dmaps(self.coord_recs, self.coord_ligs,
                self.padmask_recs, self.padmask_ligs)

        self.edge_recs, self.send_recs, self.neigh_recs = \
                vmap(partial(self.get_edges, self.edge_num))(
                        self.length_recs, self.rmaps)
        self.edge_ligs, self.send_ligs, self.neigh_ligs = \
                vmap(partial(self.get_edges, self.edge_num))(
                        self.length_ligs, self.lmaps)


    def reset(self, dataset, idxs):
        '''
        Reset cloud informations for pairs whose index is in idxs. 
        Cloud states are set back to what is stored in dataset.
        '''

        ca_clouds = []
        for idx, code in enumerate(self.list):
            if idx in idxs: ca_clouds.append(dataset[code]['coord'][1])
            else: ca_clouds.append(self.coord_ligs[idx])
        self.coord_ligs = jnp.stack(ca_clouds)


    @partial(jax.jit, static_argnums=(0,1))
    def get_edges(self, edge_num, seqlen, dmap):

        def _get_residue_edges(edge_num, line):

            def _next_lowest(val, line):
                line = jnp.where(line>val, line, 1e6)
                next_val = jnp.min(line)
                return next_val, jnp.argwhere(line==next_val, size=1)

            sorting_input = jnp.broadcast_to(line[None,:], (edge_num, line.shape[0]))
            sorted_idxs = jnp.squeeze(jax.lax.scan(_next_lowest, 0, sorting_input)[1])
            sorted_line = line[sorted_idxs]
            return sorted_line, sorted_idxs

        def _get_distance_features(dist):
            gate = jnp.where(dist!=0, 1, 0)
            scales = 1.5 ** (jnp.indices((15,))[0]+1)
            return jnp.exp(-((dist ** 2) / scales))*gate

        senders = jnp.indices((dmap.shape[0],))[0]
        senders = jnp.broadcast_to(senders[:,None], (dmap.shape[0], edge_num))
        
        dist, neighs = vmap(partial(_get_residue_edges, edge_num))(dmap)
        
        dist, senders, neighs = jnp.ravel(dist), jnp.ravel(senders), jnp.ravel(neighs),
        edges = vmap(_get_distance_features)(dist)

        senders = jnp.array(senders, dtype=jnp.uint16)
        neighs = jnp.array(jnp.where(senders<seqlen, neighs, senders), dtype=jnp.uint16)
        return edges, senders, neighs


    @partial(jax.jit, static_argnums=(0,))
    def get_state(self, coord_ligs):        

        dmaps = vmap(distances_from_coords)(
                self.coord_recs[:,:,None,:], coord_ligs[:,None,:,:])

        dmaps = vmap(lambda x, y, z: x*y[:,None]*z[None,:])(
                dmaps, self.padmask_recs, self.padmask_ligs)

        edge_int_recs, send_int_recs, neigh_int_recs = \
                vmap(partial(self.get_edges, self.edge_num))(
                        self.length_recs, dmaps)

        dmaps_t = vmap(lambda x: x.T)(dmaps)
        edge_int_ligs, send_int_ligs, neigh_int_ligs = \
                vmap(partial(self.get_edges, self.edge_num))(
                        self.length_ligs, dmaps_t)

        edge_ints = jnp.concatenate((edge_int_recs, edge_int_ligs), axis=1)

        send_ints = jnp.concatenate(
                (send_int_recs, 
                 send_int_ligs+self.size), axis=1)

        neigh_ints = jnp.concatenate(
                (neigh_int_recs+self.size, 
                 neigh_int_ligs), axis=1)

        dmaps = jnp.where(dmaps!=0., dmaps, 1e6)
        mindist_recs = vmap(lambda x:jnp.min(x, axis=1))(dmaps)
        mindist_ligs = vmap(lambda x:jnp.min(x, axis=0))(dmaps)

        intmask_recs = vmap(lambda x: 
                jnp.where((4<x)&(x<=8), 1., 0.))(mindist_recs)
        intmask_ligs = vmap(lambda x: 
                jnp.where((4<x)&(x<=8), 1., 0.))(mindist_ligs)

        rimmask_recs = vmap(lambda x: 
                jnp.where((8<x)&(x<=12), 1., 0.))(mindist_recs)
        rimmask_ligs = vmap(lambda x: 
                jnp.where((8<x)&(x<=12), 1., 0.))(mindist_ligs)

        intmask_ints = jnp.concatenate((intmask_recs,intmask_ligs), axis=1)
        rimmask_ints = jnp.concatenate((rimmask_recs,rimmask_ligs), axis=1)

        mask_ints = jnp.concatenate((intmask_ints[:,:,None], 
                                     rimmask_ints[:,:,None]), axis=2)

        return mask_ints, edge_ints, send_ints, neigh_ints


    @partial(jax.jit, static_argnums=(0,))
    def format_input_state(self, masks, edges, senders, neighs):

        padmasks = jnp.concatenate((self.padmask_recs, 
                                    self.padmask_ligs), axis=1)

        masks = jnp.concatenate((padmasks[:,:,None], masks), axis=2)

        nodes = jnp.concatenate((self.feat_recs, 
                                 self.feat_ligs), axis=1)
        
        edges = jnp.concatenate((self.edge_recs, 
                                 self.edge_ligs, 
                                 edges), axis=1)
        
        senders = jnp.concatenate((self.send_recs, 
                                   self.send_ligs, 
                                   senders), axis=1)
        
        neighs = jnp.concatenate((self.neigh_recs, 
                                  self.neigh_ligs, 
                                  neighs), axis=1)

        return masks, nodes, edges, senders, neighs


    @partial(jit, static_argnums=(0,))
    def get_random_action(self, key, mask_ints):

        def _compose_action(key, intmask_lig, rimmask_rec, rimmask_lig):
            
            key, keyn = jrn.split(key, 2)
            noise = jrn.uniform(keyn, shape=rimmask_lig.shape)
            lig_idx = jnp.argmax(rimmask_lig*noise)

            key, keyn = jrn.split(key, 2)
            noise = jrn.uniform(keyn, shape=rimmask_rec.shape)
            rec_idx = jnp.argmax(rimmask_rec*noise)

            key, keyn = jrn.split(key, 2)
            noise = jrn.uniform(keyn, shape=intmask_lig.shape)
            pivot_idx = jnp.argmax(intmask_lig*noise)

            # compose action indexes
            return jnp.array((pivot_idx, lig_idx, rec_idx))

        mask_recs, mask_ligs = jnp.split(mask_ints, 2, axis=1)
        intmask_recs, rimmask_recs = jnp.split(mask_recs, 2, axis=2)
        intmask_ligs, rimmask_ligs = jnp.split(mask_ligs, 2, axis=2)
        
        return vmap(partial(_compose_action, key))(
                intmask_ligs, rimmask_recs, rimmask_ligs)


    @partial(jit, static_argnums=(0,))
    def refine_action(self, coord_ligs, actions):
        
        coord_recs = self.coord_recs

        # get starting, target and pivot residues
        Ps = vmap(lambda x, y: x[y[0]])(coord_ligs, actions)
        p0s = vmap(lambda x, y: x[y[1]])(coord_ligs, actions)
        rec_anchors = vmap(lambda x, y: x[y[2]])(coord_recs, actions)
        p1s = vmap(lambda x, y: x+y/2)(p0s, rec_anchors)

        # normalize new lig residue position to have right distance from pivot center
        real_Pp0s = vmap(lambda x, y: jnp.linalg.norm(x-y))(p0s, Ps)
        p1s = vmap(lambda x, y, z: ((x-y)/jnp.linalg.norm(x-y)*z)+y)(p1s, Ps, real_Pp0s)

        return Ps, p0s, p1s

    
    @partial(jit, static_argnums=(0,))
    def step(self, coord_ligs, Ps, quats):

        def _single_step(cloud, P, quat, mask):
            cloud -= P[None,:]
            cloud  = quat_rotation(cloud, quat)
            cloud += P[None,:]
            return cloud*mask[:,None]

        padmask_ligs = self.padmask_ligs
        coord_ligs = vmap(_single_step)(coord_ligs, Ps, quats, padmask_ligs)

        return coord_ligs


    @partial(jit, static_argnums=(0,))
    def get_rewards(self, dmaps):
        def _get_reward(dmap, dmap_true, count):
            thr = 3
            d_val = jnp.abs(jnp.subtract(dmap, dmap_true))-3 # range 0<x<inf becomes -3<x<inf
            c_val = jnp.where((dmap>=thr)|(dmap==0), 0, dmap)-(thr/2) # range 0<x<3 becomes -1.5<x<1.5
            deltas = jax.nn.sigmoid(d_val*3)
            clashes = jax.nn.sigmoid(c_val*6)
            divisor = (4.5*deltas)+(4.5*clashes)

            interface_count = jnp.sum(jnp.where((dmap_true<=8)&(dmap_true!=0), 1, 0))
            interface_max = 50/interface_count
            int_scores = jnp.where(dmap_true<=8, interface_max, 0)
            int_scores = int_scores/(1+divisor)
            int_scores = jnp.where((dmap!=0)&(dmap_true<=8), int_scores, 0)

            all_max = 50/count
            all_scores = all_max/(1+divisor)
            all_scores = jnp.where(dmap!=0, all_scores, 0)
            return jnp.sum(int_scores+all_scores)
         
        count = self.length_recs*self.length_ligs
        return vmap(_get_reward)(dmaps, self.true_ints, count)

    def move_from_native(self, key, reset_idxs, deviation, config):
        true_coords = self.coord_ligs
        lig_RMSDs = vmap(rmsd)(true_coords, self.coord_ligs, self.length_ligs)

        while jnp.any(lig_RMSDs < deviation):
            # get state
            (mask_ints, edge_ints, i_ints, j_ints) = self.get_state(self.coord_ligs)
            
            # elaborate random action
            key, key_n = jrn.split(key, 2)
            actions = self.get_random_action(key_n, mask_ints)
            Ps, p1s, p2s = self.refine_action(self.coord_ligs, actions)
            quats = vmap(quat_from_pivoting)(Ps, p1s, p2s)
            
            # set to identity quaternions of legal interfaces matching RMSD
            active_idxs = []
            all_idxs = jnp.indices((self.number,))[0]
            done_idxs = jnp.argwhere(lig_RMSDs > deviation)
            for idx in all_idxs:
                if idx in done_idxs or idx not in reset_idxs:
                    quats = quats.at[idx].set(jnp.array((1.,0.,0.,0.)))
                else: active_idxs.append(idx)
            active_idxs = jnp.array(active_idxs)
            
            # get updated coordinates and distance map
            coord_ligs = self.step(self.coord_ligs, Ps, quats)
            dmaps = vmap(distances_from_coords)(
                    self.coord_recs[:,:,None,:], coord_ligs[:,None,:,:])
            dmaps = vmap(lambda x,y,z: x*y[:,None]*z[None,:])(
                    dmaps, self.padmask_recs, self.padmask_ligs)

            # while there is some illegal move
            while illegal_interfaces(dmaps, config, False):
                
                # new actions set
                key, key_n = jrn.split(key, 2)
                actions_new = self.get_random_action(key_n, mask_ints)
                Ps_new, p1s_new, p2s_new = self.refine_action(self.coord_ligs, actions_new)
                quats_new = vmap(quat_from_pivoting)(Ps_new, p1s_new, p2s_new)

                # set new quaternions and pivot points for illegal actions
                illegal_idxs = get_illegal_idxs(dmaps, config)
                for idx in all_idxs:
                    if idx in illegal_idxs: 
                        quats = quats.at[idx].set(quats_new[idx])
                        Ps = Ps.at[idx].set(Ps_new[idx])
                
                # update again coordinates and distance map
                coord_ligs = self.step(self.coord_ligs, Ps, quats)
                dmaps = vmap(distances_from_coords)(
                        self.coord_recs[:,:,None,:], coord_ligs[:,None,:,:])
                dmaps = vmap(lambda x,y,z: x*y[:,None]*z[None,:])(
                        dmaps, self.padmask_recs, self.padmask_ligs)
            
            self.coord_ligs = coord_ligs
            lig_RMSDs = vmap(rmsd)(true_coords, self.coord_ligs, self.length_ligs)

            # set RMSD above the requirement for idxs which has not been reset
            for idx in all_idxs:
                if idx not in reset_idxs: lig_RMSDs = lig_RMSDs.at[idx].set(deviation)
                
        print ('New starting RMSD after reset:', lig_RMSDs)

if __name__ == '__main__':
    sys.exit()
