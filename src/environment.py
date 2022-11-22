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

        def _stack_dataset(idx):
            masks = jnp.stack([dataset[pdb]['masks'][idx] for pdb in self.list])
            clouds = jnp.stack([dataset[pdb]['clouds'][idx] for pdb in self.list])     
            nodes = jnp.stack([dataset[pdb]['nodes'][idx] for pdb in self.list])
            edges = jnp.stack([dataset[pdb]['edges'][idx] for pdb in self.list])
            senders = jnp.stack([dataset[pdb]['senders'][idx] for pdb in self.list])
            receivers = jnp.stack([dataset[pdb]['receivers'][idx] for pdb in self.list])
            return masks, clouds, nodes, edges, senders, receivers

        self.m_rec, self.c_rec, self.n_rec,\
        self.e_rec, self.s_rec, self.r_rec  = _stack_dataset(0)
        self.n_rec = jnp.concatenate([self.n_rec, self.c_rec], axis=-1)

        self.m_lig, self.c_lig, self.n_lig,\
        self.e_lig, self.s_lig, self.r_lig  = _stack_dataset(1)
        self.n_lig = jnp.concatenate([self.n_lig, self.c_lig], axis=-1)

        self.e_lbl = jnp.stack([dataset[pdb]['ledges'] for pdb in self.list])
        self.s_lbl = jnp.stack([dataset[pdb]['lsenders'] for pdb in self.list])
        self.r_lbl = jnp.stack([dataset[pdb]['lreceivers'] for pdb in self.list])
        self.e_int = jnp.stack([dataset[pdb]['iedges'] for pdb in self.list])
        self.s_int = jnp.stack([dataset[pdb]['isenders'] for pdb in self.list])
        self.r_int = jnp.stack([dataset[pdb]['ireceivers'] for pdb in self.list])

        self.distances_from_cloud = vmap(distances_from_cloud)
        
        self.surf_rec = jnp.where(self.m_rec>=0.2, 1, 0)
        self.padmask_rec = jnp.where(self.m_rec!=0, 1, 0)

        self.surf_lig = jnp.where(self.m_lig>=0.2, 1, 0)
        self.padmask_lig = jnp.where(self.m_lig!=0, 1, 0)
        #self.dmaps = self.distances_from_cloud(self.c_rec[:,:,None,:],
        #                                       self.c_lig[:,None,:,:])
        #self.dmaps = vmap(lambda x, y, z: x*y[:,None]*z[None,:])(self.dmaps, m_rec, m_lig)

        self.l_rec = jnp.sum(jnp.where(self.m_rec!=0, 1, 0), axis=1)
        self.l_lig = jnp.sum(jnp.where(self.m_lig!=0, 1, 0), axis=1)


    def reset(self, dataset, idxs):

        clouds, edges, senders, receivers = [], [], [], []
        for idx, code in enumerate(self.list):
            if idx in idxs: clouds.append(dataset[code]['clouds'][1])
            else: clouds.append(self.c_lig[idx])
            if idx in idxs: edges.append(dataset[code]['edges'][1])
            else: edges.append(self.e_lig[idx])
            if idx in idxs: senders.append(dataset[code]['senders'][1])
            else: senders.append(self.s_lig[idx])
            if idx in idxs: receivers.append(dataset[code]['receivers'][1])
            else: receivers.append(self.r_lig[idx])
        self.c_lig = jnp.stack(clouds)
        self.e_lig = jnp.stack(edges)
        self.s_lig = jnp.stack(senders)
        self.r_lig = jnp.stack(receivers)
        self.n_lig = jnp.concatenate([self.n_lig[:,:,:-3], self.c_lig], axis=-1)
        #self.dmaps = self.distances_from_cloud(self.c_rec[:,:,None,:],
        #                                       self.c_lig[:,None,:,:])
        #m_rec = jnp.where(self.m_rec!=0, 1, 0)
        #m_lig = jnp.where(self.m_lig!=0, 1, 0)
        #self.dmaps = vmap(lambda x, y, z: x*y[:,None]*z[None,:])(self.dmaps, m_rec, m_lig)

    @partial(jit, static_argnums=(0,))
    def step(self, clouds, actions):

        def _single_step(cloud, action, mask, length):
            geom_center = jnp.sum(cloud, axis=0)/length
            ligand_frame = cloud - geom_center[None,:]
            transformed_cloud = quat_rotation(ligand_frame, action[:4])
            transformed_cloud = transformed_cloud + geom_center[None,:]
            return (transformed_cloud+action[None,4:-1])*mask[:,None]

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

        c_lig_new = vmap(_single_step)(clouds, actions, self.padmask_lig, self.l_lig)

        dmaps_new = self.distances_from_cloud(
                self.c_rec[:,:,None,:], c_lig_new[:,None,:,:])
        
        dmaps_new = vmap(lambda x, y, z: x*y[:,None]*z[None,:])(
                dmaps_new, self.padmask_rec, self.padmask_lig)
        
        e_int, s_int, r_int = vmap(_get_interface_graph)(dmaps_new)
        
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
