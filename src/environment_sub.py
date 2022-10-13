import sys
import jax
import time
import dill as pickle
import jax.numpy as jnp
from functools import partial
from jax import jit

from ops import *

class DockingEnv():
    def __init__(self, dataset, PNRG, substeps=10):
        self.key = PNRG
        self.list = list(dataset.keys())
        self.substeps = substeps

        self.l_rec = {len(dataset[pdb]['clouds'][0]) for pdb in dataset}
        self.l_lig = {len(dataset[pdb]['clouds'][1]) for pdb in dataset}

        self.m_rec = {pdb:jnp.where(dataset[pdb]['masks'][0]>=0.2, 0, 10) \
                for pdb in dataset}
        self.m_lig = {pdb:jnp.where(dataset[pdb]['masks'][1]>=0.2, 0, 10) \
                for pdb in dataset}
        self.g_rec = {pdb:dataset[pdb]['graphs'][0] for pdb in dataset}
        self.g_lig = {pdb:dataset[pdb]['graphs'][1] for pdb in dataset}
        self.c_rec = {pdb:dataset[pdb]['clouds'][0] for pdb in dataset}         

        self.labels = {pdb:jnp.where((cmap<8) & (cmap>=3), 1, 0) for pdb in dataset}


    def reset(self, dataset):
        self.c_lig = {pdb:dataset[pdb]['clouds'][1] for pdb in dataset}
        self.contacts = jax.tree_util.tree_map(
                lambda x, y: cmap_from_cloud(x[:,None,:], y[None,:,:]),
                self.c_rec, self.c_lig)


    @partial(jit, static_argnums=(0,))
    def sample(self,):
        '''
        sample action dataset wide
        '''
        def _sample_rot():
            rot = jax.random.uniform(
                self.key, shape=(len(self.list), self.substeps, 3),
                minval=-1, maxval=1)
            ones = jnp.ones((len(self.list), self.substeps, 1))
            rot = jnp.concatenate((rot, ones), axis=-1)
            rot = rot/jnp.sqrt(jnp.sum(rot**2, axis=-1))[:,:,None]
            return rot

        def _sample_tr():
            return jax.random.uniform(
                self.key, shape=(len(self.list), self.substeps, 3),
                minval=-10, maxval=10)
        
        dataset_step = jnp.concatenate(
            (_sample_rot(), _sample_tr()), axis=-1)

        return {pdb:step for pdb, step in zip(self.list, dataset_step)}
 
    
    @partial(jit, static_argnums=(0,))
    def step(self, actions):

        def _compose_substeps(action):

            def _quat_composition(q1, q2):
                y = quat_composition(q2, q1)
                return y, y

            def _tr_composition(tr1, tr2):
                y = tr1+tr2
                return y, y

            composed_quat = jax.lax.scan(_quat_composition(),
                jnp.array([1.,0.,0.,0.]), action[:,:4])

            composed_tr = jax.lax.scan(_tr_composition(),
                jnp.array([0.,0.,0.]), action[:,4:])

            return jnp.concatenate((composed_quat, composed_tr) axis=-1)

        def _step(actions, clouds=self.c_lig):
            
            def _one_cloud_step(cloud, action):
                rot_cloud = quat_rotation(cloud, action[:,:4], self.substeps)
                tr_cloud = rot_cloud+action[:,None,4:]
                return tr_cloud
            
            geom_centers = jax.tree_util.tree_map(
                    lambda x: jnp.sum(x, axis=0)/x.shape[0], 
                    clouds)

            ligand_frames = jax.tree_util.tree_map(
                    lambda x, y: jnp.sum((x, -y[None,:])),
                    clouds, mass_centers)

            transformed = jax.tree_util.tree_map(_one_cloud_step, 
                    ligand_frames, actions)

            return jax.tree_util.tree_map(lambda x, y: jnp.sum((x, y[None,:])),
                transformed, mass_centers))

        def _get_reward(ligs, rec, real=self.labels):
            '''
            rec shape: (None, N_res, None, 3)
            ligs shape: (substeps, None, N_res, 3)
            '''
            cmaps = cmap_from_cloud(rec[None,:,None,:], ligs[:,None,:,:])

            clash = jnp.sum(jnp.where(cmaps[:-1,:,:]<3, 1, 0))
            clash_l = jnp.sum(jnp.where(cmaps[-1,:,:]<3, 1, 0))

            cont = jnp.where((cmaps[:-1,:,:]<8) & (cmaps[:-1,:,:]>=3), 1, 0)
            cont_l = jnp.where((cmaps[-1,:,:]<8) & (cmaps[-1,:,:]>=3), 1, 0)

            i_prev, i_next = -1*cont[:-1,:,:], cont[1:,:,:]
            delta = jnp.sum(jnp.concatenate((i_prev, i_next), axis=-1))
            delta = jnp.where(delta>0, 1, 0)
            delta += delta*real[None,:,:]
            delta = jnp.sum(delta)

            match = jnp.sum(jnp.where((cont_l==1) & (real==1), 1, 0))
            mismatch = jnp.sum(jnp.where((cont_l==1) & (real==0), 1, 0))

            cont_l = jnp.sum(cont_l)
            cont = jnp.sum(cont+(cont*real[None,:,:]))

            reward = (cont+delta)-clash/(2*self.substeps)+match-mismatch-clash_l

            return [cmaps[-1,:,:], reward, cont_l]

        actions = jax.tree_util.tree_map(_compose_substeps, actions)
        states = _step(actions)
        evaluation = jax.tree_util.tree_map(_get_reward, states, self.c_rec, self.labels)
        
        self.c_lig = jax.tree_util.tree_map(lambda x: jnp.squeeze(x[-1]), states)
        self.contacts = jax.tree_util.tree_map(lambda x: x[0], evaluation)

        rewards = jax.tree_util.tree_map(lambda x: x[1], evaluation)
        status = jax.tree_util.tree_map(lambda x: x[2], evaluation)
        return rewards, status

    def get_state(self, thr=8):
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
        iedges = jax.tree_util.tree_map(one_hot_distances(), iedges)
        iedges = jax.tree_util.tree_map(
                lambda x: jnp.concatenate(
                    (x,
                     jnp.zeros((x.shape[0], 1)),
                     jnp.zeros((x.shape[0], 1)),
                     jnp.ones((x.shape[0], 1))), axis=-1), iedges)

        isender = jax.tree_util.tree_map(
                lambda x: x-self.l_rec-self.l_lig, rec_idx)
        ireceiver = jax.tree_util.tree_map(
                lambda x: x-self.l_lig, lig_idx)
        
        # non-surface/non-interface indexes
        def _get_noninterface_edges(contacts, surf_masks, 
                                    downscales1, downscales2):
            senders = jax.tree_util.tree_map(
                    lambda x, y, z: jnp.where(jnp.min(x+y[:,None], axis=1)>thr)-z,
                    contacts, surf_masks, downscales1)

            receivers = jax.tree_util.tree_map(
                    lambda x, y, z: jnp.argmin(x[y,:], axis=1)-z, 
                    contacts, senders, downscales2)
            
            edges = jax.tree_util.tree_map(
                    lambda x, y, z: x[y,z], contacts, senders, receivers)
        
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
                self.contacts.transpose(), self.m_lig, 
                self.l_lig, self.l_rec)

        redges = jax.tree_util.tree_map(one_hot_distances(), redges)
        ledges = jax.tree_util.tree_map(one_hot_distances(), ledges)

        all_edges = _concat_across_dic(iedges, redges, ledges)
        all_senders = _concat_across_dic(isenders, rsenders, lsenders)
        all_receivers = _concat_across_dic(ireceivers, rreceivers, lreceivers)
        
        return jax.tree_util.tree_map(_interface_graph(),
                all_edges, all_senders, all_receivers)

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
