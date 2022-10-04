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

        self.lengths = {}
        for pdb in dataset:
            self.lengths[pdb] = (
                len(dataset[pdb]['clouds'][0]), 
                len(dataset[pdb]['clouds'][1]))

        self.n_rec = {pdb:dataset[pdb]['nodes'][0] for pdb in dataset}
        self.n_lig = {pdb:dataset[pdb]['nodes'][1] for pdb in dataset}
        self.e_rec = {pdb:dataset[pdb]['edges'][0] for pdb in dataset}
        self.e_lig = {pdb:dataset[pdb]['edges'][1] for pdb in dataset}
        self.c_rec = {pdb:dataset[pdb]['clouds'][0] for pdb in dataset}         

        self.labels = {}
        for pdb in dataset:
            if dataset[pdb]['interface'].shape[0] == self.c_rec[pdb].shape[0]:
                cmap = dataset[pdb]['interface']
            else:
                cmap = dataset[pdb]['interface'].transpose()
            self.labels[pdb] = jnp.where((cmap<8) & (cmap>=3), 1, 0)
        

    def reset(self, dataset):
        self.c_lig = {pdb:dataset[pdb]['clouds'][1] for pdb in dataset}

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

        def _get_reward(ligs, rec, real):
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

            return jnp.array([reward, cont_l])

        actions = jax.tree_util.tree_map(_compose_substeps, actions)
        states = _step(actions)
        evaluation = jax.tree_util.tree_map(_get_reward, states, self.c_rec, self.labels)
        new_states = jax.tree_util.tree_map(lambda x: jnp.squeeze(x[-1]), states)
        rewards = jax.tree_util.tree_map(lambda x: x[0], evaluation)
        status = jax.tree_util.tree_map(lambda x: x[1], evaluation)
        return new_states, rewards, status

    def update(self, new_states):
        '''
        update actual ligand positions dataset-wide
        '''
        def _single_update(state, pdb):
            self.c_lig[pdb] = state[-1]

        jax.tree_util.tree_map(_single_update, new_states, (self.list))


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
