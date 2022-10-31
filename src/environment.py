import sys
import jax
import time
import jraph
import pickle
import jax.numpy as jnp
from functools import partial
from jax import jit

from ops import *
from graph import *

class GWrapper():
    def __init__(self, g):
        self.g = g

class DockingEnv():
    def __init__(self, dataset, PNRG):
        self.key = PNRG
        self.list = list(dataset.keys())

        self.visit_lookup = {
                pdb:jnp.zeros((dataset[pdb]['clouds'][0].shape[0],
                               dataset[pdb]['clouds'][1].shape[0])) \
                for pdb in dataset}


        self.len_rec = {pdb:dataset[pdb]['clouds'][0].shape[0] for pdb in dataset}
        self.len_lig = {pdb:dataset[pdb]['clouds'][1].shape[0] for pdb in dataset}

        self.smask_rec = {pdb:dataset[pdb]['smasks'][0] for pdb in dataset}
        self.smask_lig = {pdb:dataset[pdb]['smasks'][1] for pdb in dataset}
        
        self.bmask_rec = {pdb:dataset[pdb]['bmasks'][0] for pdb in dataset}
        self.bmask_lig = {pdb:dataset[pdb]['bmasks'][1] for pdb in dataset}
        
        self.graph_rec = {pdb:dataset[pdb]['graphs'][0] for pdb in dataset}
        self.graph_lig = {pdb:dataset[pdb]['graphs'][1] for pdb in dataset}

        self.cloud_rec = {pdb:dataset[pdb]['clouds'][0] for pdb in dataset}         
        self.scloud_rec = self.mask_filter(self.cloud_rec, self.smask_rec)
        self.bcloud_rec = self.mask_filter(self.cloud_rec, self.bmask_rec)

        self.labels = {pdb:jnp.where(
            (dataset[pdb]['interface']<8) & (dataset[pdb]['interface']>=3), 
            1, 0) for pdb in dataset}


    def reset(self, dataset):
        self.cloud_lig = {pdb:dataset[pdb]['clouds'][1] for pdb in dataset}
        self.scloud_lig = self.mask_filter(self.cloud_lig, self.smask_lig)
        self.bcloud_lig = self.mask_filter(self.cloud_lig, self.bmask_lig)
        
        self.contacts = jax.tree_util.tree_map(
                lambda x, y: cmap_from_cloud(x[:,None,:], y[None,:,:]),
                self.cloud_rec, self.cloud_lig)

    def mask_filter(self, cloud, mask):
        return jax.tree_util.tree_map(lambda x, y: x[y], cloud, mask)

    @partial(jit, static_argnums=(0,))
    def step(self, actions):

        def _step(actions, clouds):
            
            def _one_cloud_step(cloud, action):
                rot_cloud = quat_rotation(cloud, action[:4])
                tr_cloud = rot_cloud+action[None,4:-1]
                return tr_cloud
            
            geom_centers = jax.tree_util.tree_map(
                    lambda x: jnp.sum(x, axis=0)/x.shape[0], 
                    clouds)

            ligand_frames = jax.tree_util.tree_map(lambda x, y: x-y[None,:],
                    clouds, geom_centers)
            

            transformed = jax.tree_util.tree_map(_one_cloud_step, 
                    ligand_frames, actions)

            return jax.tree_util.tree_map(lambda x, y: x+y[None,:],
                    transformed, geom_centers)

        
        def _get_reward(new_cmap, new_cont, real, diff, lookup, action):
            confidence = action[-1]
            
            clashes = jnp.sum(jnp.where(new_cmap<3, 1, 0))

            diff = jnp.sum(diff*jnp.clip(
                        1/2**(8+lookup), a_min=0.00001, a_max=0.001))

            match = jnp.sum(jnp.where((new_cont==1) & (real==1), 1, 0))
            mismatch = jnp.sum(jnp.where((new_cont==1) & (real==0), 1, 0))
            
            final_reward = (match-(mismatch+clashes))*confidence
            inner_reward = diff-(clashes*(1-confidence))-jnp.min(new_cmap)
            return inner_reward+final_reward

        new_ligs = _step(actions, self.cloud_lig)
        new_cmaps = jax.tree_util.tree_map(
                lambda x, y: cmap_from_cloud(x[:,None,:], y[None,:,:]), self.cloud_rec, new_ligs)
        
        old_cont = jax.tree_util.tree_map(
                lambda x: jnp.where((x<8) & (x>=3), 1, 0), self.contacts)
        new_cont = jax.tree_util.tree_map(
                lambda x: jnp.where((x<8) & (x>=3), 1, 0), new_cmaps)
        diff_contacts = jax.tree_util.tree_map(
                lambda x, y: jnp.where((x==1) & (y==0), 1, 0), new_cont, old_cont)
        updated_lookup = jax.tree_util.tree_map(
                lambda x, y: x+y, self.visit_lookup, diff_contacts)

        reward = jax.tree_util.tree_map(
                _get_reward, new_cmaps, new_cont, self.labels, 
                diff_contacts, self.visit_lookup, actions)
        
        return reward, new_ligs, new_cmaps, updated_lookup

    @partial(jit, static_argnums=(0,))
    def get_state(self):

        def _concat_across_dic(sr, sl, br, bl):
            return jax.tree_util.tree_map(
                lambda a,b,c,d: jnp.concatenate((a,b,c,d), axis=0), sr, sl, br, bl)

        def _interface_graph(edges, senders, receivers):
            n_edges = jnp.array(senders.shape[0])
            return GWrapper(jraph.GraphsTuple(
                        nodes=jnp.array([]), edges=edges,
                        senders=senders, receivers=receivers,
                        n_node=jnp.array([0]), n_edge=n_edges,
                        globals=[1,1]))

        #s-urface cmaps and b-uried cmaps
        scmaps = jax.tree_util.tree_map(
                lambda x, y: cmap_from_cloud(x[:,None,:], y[None,:,:]), 
                self.scloud_rec, self.scloud_lig)
        bcmaps = jax.tree_util.tree_map(
                lambda x, y: cmap_from_cloud(x[:,None,:], y[None,:,:]),
                self.bcloud_rec, self.bcloud_lig)
        
        # compute surface edges
        surface_links_r = jax.tree_util.tree_map(surface_edges, scmaps, self.smask_rec)
        surface_links_l = jax.tree_util.tree_map(
                lambda x, y: surface_edges(x.transpose(), y), scmaps, self.smask_lig)
        
        # unpack surface edges and indexes
        sedges_r = jax.tree_util.tree_map(lambda x: x[0], surface_links_r)
        ssenders_r = jax.tree_util.tree_map(lambda x: x[1], surface_links_r)
        sreceivers_r = jax.tree_util.tree_map(lambda x: x[2], surface_links_r)
        sedges_l = jax.tree_util.tree_map(lambda x: x[0], surface_links_l)
        ssenders_l = jax.tree_util.tree_map(lambda x: x[1], surface_links_l)
        sreceivers_l = jax.tree_util.tree_map(lambda x: x[2], surface_links_l)

        #compute buried edges
        breceivers_r = jax.tree_util.tree_map(lambda x: jnp.argmin(x, axis=1), bcmaps)
        breceivers_l = jax.tree_util.tree_map(lambda x: jnp.argmin(x, axis=0), bcmaps)
        bsenders_r = jax.tree_util.tree_map(lambda x: jnp.indices(x.shape)[0], breceivers_r)
        bsenders_l = jax.tree_util.tree_map(lambda x: jnp.indices(x.shape)[0], breceivers_l)
        bedges_r = jax.tree_util.tree_map(lambda x, y, z: x[y,z], bcmaps, bsenders_r, breceivers_r)
        bedges_l = jax.tree_util.tree_map(lambda x, y, z: x[y,z], bcmaps, bsenders_l, breceivers_l)

        # shift to full cmap indexes
        bsenders_r = jax.tree_util.tree_map(lambda x, y: x[y], self.bmask_rec, bsenders_r)
        bsenders_l = jax.tree_util.tree_map(lambda x, y: x[y], self.bmask_lig, bsenders_l)
        breceivers_r = jax.tree_util.tree_map(lambda x, y: x[y], self.bmask_lig, breceivers_r)
        breceivers_l = jax.tree_util.tree_map(lambda x, y: x[y], self.bmask_rec, breceivers_l)

        # apply shift to receptor indexes to match pre-existing graph indexes
        rec_shift = self.len_rec+self.len_lig
        ssenders_r -= rec_shift
        bsenders_r -= rec_shift
        sreceivers_l -= rec_shift
        breceivers_l -= rec_shift
        
        # apply shift to ligand indexes to match pre-existing graph indexes
        ssenders_l -= self.len_lig
        bsenders_l -= self.len_lig
        sreceivers_r -= self.len_lig
        breceivers_r -= self.len_lig

        # encode edges features
        sedges_r = jax.tree_util.tree_map(encode_distances, sedges_r)
        sedges_l = jax.tree_util.tree_map(encode_distances, sedges_l)
        bedges_r = jax.tree_util.tree_map(encode_distances, bedges_r)
        bedges_l = jax.tree_util.tree_map(encode_distances, bedges_l)

        # concatenate edges
        all_edges = _concat_across_dic(sedges_r, sedges_l, bedges_r, bedges_l)
        all_senders = _concat_across_dic(ssenders_r, ssenders_l, bsenders_r, bsenders_l)
        all_receivers = _concat_across_dic(sreceivers_r, sreceivers_l, breceivers_r, breceivers_l)
        
        return jax.tree_util.tree_map(_interface_graph, all_edges, all_senders, all_receivers)

def main():
    data_path = '/home/pozzati/complex_assembly/data/train_features.pkl'
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)

    dataset = {pdb:dataset[pdb] for n, pdb in enumerate(dataset) if n < 500}
    print (len(list(dataset.keys())),'examples')
    
    env = DockingEnv(dataset, 42)
    env.reset(dataset)

    print (sys.getsizeof(env.c_lig))
    print (dataset.keys())

    action = jnp.array([10*[]])

    t1 = time.perf_counter()
    action = env.sample()
    t2 = time.perf_counter()
    print ('Time to sample 1:', t2-t1)

    t1 = time.perf_counter()
    s1, r, st = env.step(action)
    t2 = time.perf_counter()
    print ('Step 0 time:', t2-t1)

    for n in range(10):
        t1 = time.perf_counter()
        s, r, st = env.step(action)
        t2 = time.perf_counter()
        print (f'Step {n+1} time:', t2-t1)


if __name__ == '__main__':
    main()
