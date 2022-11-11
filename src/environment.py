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
        self.list = list(dataset.keys())
        self.size = dataset[self.list[0]][0].shape[0]
        self.number = len(self.list)
        self.init_rt = {pdb:dataset[pdb]['init_rt'] for pdb in self.list}
        self.enum = enum
        self.pad = pad

        def _stack_dataset(idx):
            masks = jnp.stack([dataset[pdb]['masks'][idx] for pdb in self.list])
            clouds = jnp.stack([dataset[pdb]['clouds'][idx] for pdb in self.list])     
            nodes = jnp.stack([dataset[pdb]['nodes'][idx] for pdb in self.list])
            edges = jnp.stack([dataset[pdb]['edges'][idx] for pdb in self.list])
            iedges = jnp.stack([dataset[pdb]['iedges'][idx] for pdb in self.list])
            senders = jnp.stack([dataset[pdb]['senders'][idx] for pdb in self.list])
            isenders = jnp.stack([dataset[pdb]['isenders'][idx] for pdb in self.list])
            receivers = jnp.stack([dataset[pdb]['receivers'][idx] for pdb in self.list])
            ireceivers = jnp.stack([dataset[pdb]['ireceivers'][idx] for pdb in self.list])
            return masks, clouds, nodes, edges, senders, receivers, 
                   iedges, isenders, ireceivers

        self.masks_rec, self.clouds_rec, self.nodes_rec,\
        self.edges_rec, self.senders_rec, self.receivers_rec,\
        self.iedges_rec, self.isenders_rec, self.ireceivers_rec = _stack_dataset(0)

        self.masks_lig, self.clouds_lig, self.nodes_lig,\
        self.edges_lig, self.senders_lig, self.receivers_lig,\
        self.iedges_lig, self.isenders_lig, self.ireceivers_lig = _stack_dataset(1)

        self.distances_from_cloud = vmap(distances_from_cloud)
        self.dmaps = self.distances_from_cloud(self.clouds_rec[:,None,:],
                                               self.clouds_lig[None,:,:])

        self.lengths_rec = jnp.sum(jnp.where(self.mask_rec!=0, 1, 0), axis=0)
        self.lengths_lig = jnp.sum(jnp.where(self.mask_lig!=0, 1, 0), axis=0)


    def reset(self, dataset):
        self.clouds_lig = jnp.stack([dataset[pdb]['clouds'][idx] for pdb in dataset])
        self.cmaps = self.distances_from_cloud(self.clouds_rec[:,None,:],
                                               self.clouds_lig[None,:,:])

    @partial(jit, static_argnums=(0,))
    def step(self, clouds, actions):

        def _single_step(cloud, action, length):
            geom_center = jnp.sum(cloud)/length
            ligand_frames = cloud - geom_center[None,:]
            transformed_cloud = quat_rotation(ligand_frames, action[:4])
            transformed_cloud = transformed_cloud+action[4:-1]
            return transformed_cloud + geom_centers[None,:]
        
        def _get_reward(new_dmap):
            clashes = jnp.where(new_dmap<6 & new_dmap!=0, (6-new_dmap)/6, 0)
            clashes = jnp.sum(clashes)
            new_dmap = jnp.where(new_dmap==0, 1e9, new_dmap)
            return -(clashes+jnp.min(new_dmap))

        new_clouds_lig = vmap(_single_step)(clouds, actions, self.lengths_lig)
        
        new_dmaps = self.distances_from_cloud(self.clouds_rec[:,None,:], 
                                              new_clouds_lig[None,:,:])

        rewards = vmap(_get_reward)(new_dmaps)
        
        return rewards, new_clouds_lig, new_dmaps

    @partial(jit, static_argnums=(0,))
    def get_states(self, dmap):

        def _get_state(dmap, enum, pad):
            edges12, senders12, receivers12,\
            edges21, senders21, receivers21 = get_interface_edges(dmap, enum, pad) 
            all_edges = jnp.concatenate((edges12, edges21), axis=0)
            all_senders = jnp.concatenate((senders12, senders21), axis=0)
            all_receivers = jnp.concatenate((receivers12, receivers21), axis=0)
            return all_edges, all_senders, all_receivers

        return vmap(_get_state)(dmap, self.enum, self.pad, in_axes=(0,None,None))
         

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
