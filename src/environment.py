import sys
import jax
import time
import jraph
import pickle
import jax.numpy as jnp
from functools import partial
from jax import jit

from ops import *

class DockingEnv():
    def __init__(self, dataset, PNRG):
        self.key = PNRG
        self.list = list(dataset.keys())

        self.visit_lookup = {
                pdb:jnp.zeros((dataset[pdb]['clouds'][0].shape[0],
                               dataset[pdb]['clouds'][1].shape[0])) \
                for pdb in dataset}


        self.l_rec = {pdb:dataset[pdb]['clouds'][0].shape[0] for pdb in dataset}
        self.l_lig = {pdb:dataset[pdb]['clouds'][1].shape[0] for pdb in dataset}

        self.m_rec = {pdb:jnp.where(dataset[pdb]['masks'][0]>=0.2, 0, 10) \
                for pdb in dataset}
        self.m_lig = {pdb:jnp.where(dataset[pdb]['masks'][1]>=0.2, 0, 10) \
                for pdb in dataset}
        self.g_rec = {pdb:dataset[pdb]['graphs'][0] for pdb in dataset}
        self.g_lig = {pdb:dataset[pdb]['graphs'][1] for pdb in dataset}
        self.c_rec = {pdb:dataset[pdb]['clouds'][0] for pdb in dataset}         

        self.labels = {pdb:jnp.where(
            (dataset[pdb]['interface']<8) & (dataset[pdb]['interface']>=3), 
            1, 0) for pdb in dataset}


    def reset(self, dataset):
        self.c_lig = {pdb:dataset[pdb]['clouds'][1] for pdb in dataset}
        self.contacts = jax.tree_util.tree_map(
                lambda x, y: cmap_from_cloud(x[:,None,:], y[None,:,:]),
                self.c_rec, self.c_lig)


    @partial(jit, static_argnums=(0,))
    def step(self, actions, confidence):

        def _step(actions, clouds):
            
            def _one_cloud_step(cloud, action):
                rot_cloud = quat_rotation(cloud, action[:4], self.substeps)
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

        
        def _get_reward(new_lig, old_lig, rec, real, table, confidence):
            old_cmap = cmap_from_cloud(rec[:,None,:], old_lig[None,:,:])
            new_cmap = cmap_from_cloud(rec[:,None,:], new_lig[None,:,:])

            clashes = jnp.sum(jnp.where(new_cmaps<3, 1, 0))

            old_cont = jnp.where((old_cmaps<8) & (old_cmaps>=3), 1, 0)
            new_cont = jnp.where((new_cmaps<8) & (new_cmaps>=3), 1, 0)
            delta = jnp.where((new_cmaps==1) & (old_cmaps==0), 1, 0)
            table += delta
            delta = jnp.sum(delta*jnp.clip(
                        1/2**(8+table), a_min=0.00001, a_max=0.001))

            match = jnp.sum(jnp.where((new_cont==1) & (real==1), 1, 0))
            mismatch = jnp.sum(jnp.where((new_cont==1) & (real==0), 1, 0))
            
            final_reward = (match-(mismatch+clashes))*confidence
            inner_reward = delta-(clashes*(1-confidence))-jnp.min(new_cmap)
            return [inner_reward+final_reward, new_cmap, table]

        new_ligs = _step(actions, self.c_lig)
        evaluation = jax.tree_util.tree_map(
                _get_reward, new_ligs, self.c_lig, self.c_rec, 
                self.labels, self.visit_lookup, confidence)
        
        self.c_lig = jax.tree_util.tree_map(lambda x: x, new_ligs)
        self.contacts = jax.tree_util.tree_map(lambda x: x[1], evaluation)
        self.visit_lookup = jax.tree_util.tree_map(lambda x: x[2], evaluation)

        return jax.tree_util.tree_map(lambda x: x[0], evaluation)

    def get_state(self, thr=8):

        i_sign = jnp.array((0,0,1))

        # surface/interface mask
        surf_masks = jax.tree_util.tree_map(
                lambda x, y: x[:,None]+y[None,:], self.m_rec, self.m_lig)

        # surface/interface indexes
        edge_idx = jax.tree_util.tree_map(
                lambda x, y: jnp.argwhere(x+y<=thr), self.contacts, surf_masks)
        rec_idx = jax.tree_util.tree_map(lambda x: x[0], edge_idx)
        lig_idx = jax.tree_util.tree_map(lambda x: x[1], edge_idx)
        
        iedges = jax.tree_util.tree_map(
                lambda x, y, z: x[y,z], self.contacts, rec_idx, lig_idx)
        iedges = jax.tree_util.tree_map(one_hot_distances, iedges)
        iedges = jax.tree_util.tree_map(
                lambda x: jnp.concatenate((x, jnp.broadcast_to(i_sign, (x.shape[0], 3))), axis=-1), iedges)

        isenders = jax.tree_util.tree_map(
                lambda x, y, z: x-y-z, 
                rec_idx, self.l_rec, self.l_lig)
        
        ireceivers = jax.tree_util.tree_map(
                lambda x, y: x-y, lig_idx, self.l_lig)
        
        # non-surface/non-interface indexes
        def _get_noninterface_edges(contacts, surf_masks, 
                                    downscales1, downscales2, reverse=False):
            if reverse:
                contacts = jax.tree_util.tree_map(lambda x: x.transpose(), contacts)

            senders = jax.tree_util.tree_map(
                    lambda x, y: jnp.ravel(jnp.argwhere(jnp.min(x+y[:,None], axis=1)>thr)),
                    contacts, surf_masks)

            receivers = jax.tree_util.tree_map(
                    lambda x, y: jnp.argmin(x[y,:], axis=1), 
                    contacts, senders)

            edges = jax.tree_util.tree_map(lambda x, y, z: x[y,z], contacts, senders, receivers)

            senders = jax.tree_util.tree_map(lambda x, y: x-y, senders, downscales1)
            receivers = jax.tree_util.tree_map(lambda x, y: x-y, receivers, downscales2)

            return edges, senders, receivers
        
        def _concat_across_dic(i, r, l):
            return jax.tree_util.tree_map(
                lambda x,y,z: jnp.concatenate((x,y,z), axis=0), i, r, l)

        def _interface_graph(edges, senders, receivers):
            n_edges = jnp.array(senders.shape[0])
            return jraph.GraphsTuple(
                    nodes=jnp.array([]), edges=edges,
                    senders=senders, receivers=receivers,
                    n_node=jnp.array([0]), n_edge=n_edges,
                    globals=[1,1])

        redges, rsenders, rreceivers = _get_noninterface_edges(
                self.contacts, self.m_rec, 
                self.l_rec, self.l_lig)
        ledges, lsenders, lreceivers = _get_noninterface_edges(
                self.contacts, self.m_lig, 
                self.l_lig, self.l_rec, reverse=True)

        redges = jax.tree_util.tree_map(one_hot_distances, redges)
        redges = jax.tree_util.tree_map(
                lambda x: jnp.concatenate((x, jnp.broadcast_to(i_sign, (x.shape[0], 3))), axis=-1), redges)

        ledges = jax.tree_util.tree_map(one_hot_distances, ledges)
        ledges = jax.tree_util.tree_map(
                lambda x: jnp.concatenate((x, jnp.broadcast_to(i_sign, (x.shape[0], 3))), axis=-1), ledges)

        all_edges = _concat_across_dic(iedges, redges, ledges)
        all_senders = _concat_across_dic(isenders, rsenders, lsenders)
        all_receivers = _concat_across_dic(ireceivers, rreceivers, lreceivers)
        
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
