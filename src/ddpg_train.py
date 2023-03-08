# derived from https://docs.cleanrl.dev/rl-algorithms/ddpg
import os
import time
import glob
import json
import random
import argparse
import numpy as np
import pickle as pkl
import seaborn as sb
import matplotlib.pyplot as plt
from functools import partial
from gpuinfo.nvidia import get_gpus
from distutils.util import strtobool
from typing import Sequence, Iterable, Mapping, NamedTuple, Tuple

import jax
import optax
import haiku as hk
import replay_buffer
import jax.numpy as jnp
import jax.random as jrn
from jax import tree_util
from jraph import GraphsTuple
from networks.critic import Critic
from networks.actor import Actor
from replay_buffer import *
from environment import *
from ops import *

from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.Structure import Structure

def actor_fw(mask, nodes, edges, i, j):
    actor = Actor(config)
    return actor(mask, nodes, edges, i, j)

def critic_fw(mask, nodes, edges, i, j, action):
    critic = Critic(config)
    return critic(mask, nodes, edges, i, j, action)

class TrainState(NamedTuple):
    params: hk.Params
    target_params: hk.Params
    opt_state: optax.OptState

if __name__ == "__main__":    
    
    ####################################################
    # DATA READING
    ####################################################

    # load config and dataset
    path = os.getcwd()+'/'+'/'.join(__file__.split('/')[:-2])

    s = time.time()
    with open(path+'/src/config.json') as j: config = json.load(j)

    d_size, td_size = config['training_pairs'], config['test_pairs']
    dataset, code = load_dataset(path+'/data/dataset_features/', size=d_size)
    test, test_code = load_dataset(path+'/data/dataset_features/', size=td_size, skip=d_size)

    ####################################################
    # ENVIRONMENTS INITIALIZATION
    ####################################################
    
    # devices
    cpus = jax.devices('cpu')
    gpus = jax.devices('gpu')
    pmap_maxsize, pmap_devices = get_pmap_details()

    # seeding
    random.seed(config['random_seed'])
    key = jrn.PRNGKey(config['random_seed'])
    key, akey, ckey = jrn.split(key, 3)

    # environment setup
    envs = DockingEnv(dataset, config)
    tenvs = DockingEnv(test, config)
    
    # get a formatted state of training data to init parameters
    (mask_ints, edge_ints, i_ints, j_ints) = envs.get_state(envs.coord_ligs)

    ps = pmap_maxsize
    all_idxs = jnp.indices((envs.number,))[0]
    (mask, nodes, edges, i, j) = \
            envs.format_input_state(
                    mask_ints, edge_ints, i_ints, j_ints, all_idxs, 1, 1)

    # replay buffer setup
    r_buffer = ReplayBuffer(config, envs.list)

    ####################################################
    # TRAINING OBJECTS INITIALIZATION
    ####################################################

    # actor/critic models setup
    actor = hk.transform(actor_fw)
    critic = hk.transform(critic_fw)
   
    # single action agent compile
    a_apply_s = jax.jit(actor.apply)

    # batch-splitting multi-GPU agent compile
    a_apply = jax.pmap(actor.apply, 
            devices=pmap_devices)

    # batch-splitting multi-GPU critic compile
    c_apply = jax.pmap(critic.apply,
            devices=pmap_devices)

    # actor parameters initialization
    a_params = actor.init(akey, 
            mask[0], nodes[0], edges[0], i[0], j[0])
    
    # critic parameters initialization
    action = a_apply_s(a_params, None, 
            mask[0], nodes[0], edges[0], i[0], j[0])

    c_params = critic.init(ckey, 
            mask[0], nodes[0], edges[0], i[0], j[0], action)

    # duplicate and group actor/critic params, init optimizer
    a_optimiser = optax.adam(learning_rate=config['learning_rate'])
    c_optimiser = optax.adam(learning_rate=config['learning_rate']*10)

    actor_state = TrainState(
        params=a_params,
        target_params=a_params,
        opt_state=a_optimiser.init(a_params))

    critic_state = TrainState(
        params=c_params,
        target_params=c_params,
        opt_state=c_optimiser.init(c_params))

    ####################################################
    # TRAINING SET THRESHOLD RANDOMIZATION
    ####################################################

    # move ligands from native pose
    key, nkey = jrn.split(key, 2)
    original = envs.coord_ligs
    envs.move_from_native(nkey, all_idxs, config['native_dev'])
    
    dmaps = vmap(distances_from_coords)(
                envs.coord_recs[:,:,None,:], envs.coord_ligs[:,None,:,:])
    dmaps = vmap(lambda x,y,z: x*y[:,None]*z[None,:])(
                dmaps, envs.padmask_recs, envs.padmask_ligs)
    old_rewards, _, _ = envs.get_rewards(dmaps)

    rmsd = vmap(rmsd)
    old_rmsds = rmsd(original, envs.coord_ligs, envs.length_ligs)

    # get state
    (mask_ints, edge_ints, i_ints, j_ints) = envs.get_state(envs.coord_ligs)

    print ('Initialization completed! -', time.time()-s)

    ###########################################################
    # UPDATE ACCESSORY FUNCTIONS
    ###########################################################

    # function to broadcast parameters to GPU number
    params_cast = jax.jit(partial(
            jax.tree_util.tree_map, lambda x: jnp.stack([x]*ps)))

    # function to compute mean squared error and mean
    get_mse = jax.jit(
            lambda x, y: jnp.mean((y - jnp.ravel(x))**2))

    get_mean = jax.jit(lambda x: jnp.mean(-x))

    # function to compute target q value
    get_target_reward = jax.jit(
            lambda x, y: jnp.ravel(x) + (config['gamma']*jnp.ravel(y)))

    # function to update parameters
    @jax.jit
    def update_actor_params(grads, state):
        updates, new_opt_state = a_optimiser.update(grads, state.opt_state)
        return state._replace(
                params=optax.apply_updates(state.params, updates),
                opt_state=new_opt_state)
    
    @jax.jit
    def update_critic_params(grads, state):
        updates, new_opt_state = c_optimiser.update(grads, state.opt_state)
        return state._replace(
                params=optax.apply_updates(state.params, updates),
                opt_state=new_opt_state)

    ##########################################################
    # UPDATE CRITIC FUNCTION
    ##########################################################

    # function to update critic online neural network
    def update_critic(
            actor_state, critic_state,
            masks_P, nodes_P, edges_P, i_P, j_P,
            masks_N, nodes_N, edges_N, i_N, j_N,
            actions, rewards):

        # critic neural network loss function
        def crit_loss_fn(params):    
            params = params_cast(params)
            q = c_apply(params, None,
                    masks_P, nodes_P, edges_P, i_P, j_P, actions)

            return get_mse(q, y)

        # broadcast target net parameters to GPU number
        a_params = params_cast(actor_state.target_params)
        c_params = params_cast(critic_state.target_params)
        
        # compute next-step q value
        actions = a_apply(a_params, None,
                masks_N, nodes_N, edges_N, i_N, j_N)

        next_q = c_apply(c_params, None,
                masks_N, nodes_N, edges_N, i_N, j_N, actions)

        # compute target rewards
        y = get_target_reward(rewards, next_q)
        
        # compute critic gradients
        crit_loss, grads = jax.value_and_grad(crit_loss_fn)(critic_state.params)
        
        # update critic online parameters
        critic_state = update_critic_params(grads, critic_state)

        return critic_state, crit_loss

    ##############################################################
    # UPDATE ACTOR AND TARGET NETWORK FUNCTION
    ##############################################################

    def update_actor(
            actor_state, critic_state,
            masks_P, nodes_P, edges_P, i_P, j_P):
    
        def actor_loss_fn(params, c_params):
            params = params_cast(params)
            actions = a_apply(params, None,
                    masks_P, nodes_P, edges_P, i_P, j_P)

            c_params = params_cast(c_params)
            q = c_apply(c_params, None,
                    masks_P, nodes_P, edges_P, i_P, j_P, actions)

            return get_mean(q)

        # compute actor gradients
        actor_loss, grads = jax.value_and_grad(actor_loss_fn)(
                actor_state.params, critic_state.params)
        
        # update actor online parameters
        actor_state = update_actor_params(grads, actor_state)
    
        # update target parameters
        actor_state = actor_state._replace(
            target_params=optax.incremental_update(
                actor_state.params,
                actor_state.target_params,
                config['tau']))
    
        critic_state = critic_state._replace(
            target_params=optax.incremental_update(
                critic_state.params,
                critic_state.target_params,
                config['tau']))
    
        return actor_state, critic_state, actor_loss  

    ###########################################################
    # TRAINING CYCLE
    ###########################################################

    sg = time.time()
    loss_series = []
    reward_series = []
    last_iter_reset = []
    delta_series = {'labels':[], 'rmsds':[], 'rewards':[]}
    pair_reset_count = jnp.empty((envs.number,))
    
    print ('Start training! - Filling buffer...')
    for global_step in range(config['steps']):
        
        if global_step == config['buffer_size']: 
            print (f'Buffer full! - {time.time()-sg}')
            sg = time.time()

        # format state
        (masks, nodes, edges, i, j) = \
                envs.format_input_state(
                        mask_ints, edge_ints, i_ints, j_ints, all_idxs, 1, 1)

        # generate agent action
        a_params = actor_state.params
        actions = a_apply_s(actor_state.params, None, 
                masks[0], nodes[0], edges[0], i[0], j[0])
        actions = jnp.squeeze(actions)

        # elaborate action to apply to environment
        actions_t = vmap(lambda x: jnp.argmax(x, axis=1))(actions)
        Ps, p1s, p2s = envs.refine_action(envs.coord_ligs, actions_t)
        quats = vmap(quat_from_pivoting)(Ps, p1s, p2s)

        ### add noise to action

        # apply action
        coord_ligs = envs.step(envs.coord_ligs, Ps, quats)
        rmsds = rmsd(coord_ligs, envs.coord_ligs, envs.length_ligs)
        envs.coord_ligs = coord_ligs

        dmaps = vmap(distances_from_coords)(
                envs.coord_recs[:,:,None,:], envs.coord_ligs[:,None,:,:])
        dmaps = vmap(lambda x,y,z: x*y[:,None]*z[None,:])(
                dmaps, envs.padmask_recs, envs.padmask_ligs)
        
        # get reward for last action
        rewards, d_r, c_r = envs.get_rewards(dmaps)
        #print ('Full',rewards,'distance',d_r,'clashes',c_r)

        # get new state
        (mask_ints_N, edge_ints_N, i_ints_N, j_ints_N) = envs.get_state(envs.coord_ligs)
        mask_ints = jnp.array(mask_ints, dtype=jnp.uint8)
        mask_ints_N = jnp.array(mask_ints_N, dtype=jnp.uint8)
        
        # store experiences in replay buffer
        experience = [mask_ints, edge_ints, i_ints, j_ints,
                mask_ints_N, edge_ints_N, i_ints_N, j_ints_N,
                actions, rewards]

        sg = time.time()
        # buffer update
        if global_step <= config['buffer_size'] \
        and r_buffer.cache_actual < r_buffer.cache_size:
            # fast storage into a reduced size cache on GPU
            r_buffer.cache, r_buffer.cache_actual = \
                    r_buffer.add_experience(r_buffer.cache, experience, r_buffer.cache_actual)

        elif global_step <= config['buffer_size'] \
        and r_buffer.cache_actual == r_buffer.cache_size:
            
            # when cache is full move it to CPU
            r_buffer.cache = jax.device_put(r_buffer.cache, device=jax.devices('cpu')[0])
            
            # slow update of full buffer stored on CPU
            r_buffer.buff, r_buffer.cache, buff_actual = \
                    r_buffer.add_to_buffer(r_buffer.buff, r_buffer.cache, r_buffer.buff_actual)
            
            # move empty cache back to GPU
            r_buffer.cache = jax.device_put(r_buffer.cache, device=jax.devices('gpu')[0])
            
            # update count of examples in cache and buffer
            r_buffer.buff_actual = min(buff_actual, r_buffer.buff_size)
            r_buffer.cache_actual = 0

        elif global_step > config['buffer_size']:

            # move experience to CPU
            experience = jax.device_put(experience, device=jax.devices('cpu')[0])
            
            # update buffer with example
            r_buffer.buff, _ = \
                r_buffer.add_experience(r_buffer.buff, experience, r_buffer.buff_actual)
        
        print (f'Buffer update done! - {time.time()-sg}')

        sg = time.time()
        # update state for next iteration
        mask_ints = mask_ints_N
        edge_ints = edge_ints_N
        i_ints = i_ints_N
        j_ints = j_ints_N

        # check reset conditions
        for n in range(envs.number):
            increment = pair_reset_count[n]+1
            pair_reset_count = pair_reset_count.at[n].set(increment)
        
        reset_idxs = get_illegal_idxs(dmaps, config, reset=pair_reset_count)
        
        # reset selected pairs
        if len(reset_idxs) != 0:
            for n in reset_idxs:
                pair_reset_count = pair_reset_count.at[n].set(0)
            
            #print ('To reset: ', reset_idxs)
            envs.reset(dataset, reset_idxs)
            key, nkey = jrn.split(key, 2)
            envs.move_from_native(nkey, reset_idxs, config['native_dev'])
            last_iter_reset = reset_idxs

        else: last_iter_reset = []
        print (f'Reset done! - {time.time()-sg}')

        sg = time.time()
        ############# save deltas ###############
        delta_rewards = rewards - old_rewards
        delta_rmsds = rmsds - old_rmsds
        delta_rewards = jnp.ravel(delta_rewards)
        delta_rmsds = jnp.ravel(delta_rmsds)
        for n in range(rewards.shape[0]):
            if n in last_iter_reset: continue
            delta_series['rewards'].append(float(delta_rewards[n]))
            delta_series['rmsds'].append(float(delta_rmsds[n]))
            delta_series['labels'].append(envs.list[n])

        old_rmsds = rmsds
        old_rewards = rewards
        #########################################
        print (f'debug done! - {time.time()-sg}')

        # network updates
        if global_step+1 > config['buffer_size'] \
        and (global_step+1) % config['update_frequency'] == 0:
            
            # sample example batch
            key, nkey = jrn.split(key, 2)
            batch, pair_idxs = r_buffer.sample_from_buffer(nkey, r_buffer.buff)    
            batch = [jax.device_put(array, device=gpus[1]) for array in batch]
            pair_idxs = jax.device_put(pair_idxs, device=gpus[1])

            (mask_ints_P, edge_ints_P, i_ints_P, j_ints_P,
             mask_ints_N, edge_ints_N, i_ints_N, j_ints_N,
             actions, rewards) = batch

            # format batch
            bs, ps = config['batch_size_num'], pmap_maxsize
            masks_P, nodes_P, edges_P, i_P, j_P = envs.format_input_state(
                    mask_ints_P, edge_ints_P, i_ints_P, j_ints_P, pair_idxs, bs, ps)

            masks_N, nodes_N, edges_N, i_N, j_N = envs.format_input_state(
                    mask_ints_N, edge_ints_N, i_ints_N, j_ints_N, pair_idxs, bs, ps)
            
            bp = len(pair_idxs)
            new_shape = (ps, int(bs*bp/ps),) + actions.shape[-2:]
            actions = jnp.reshape(actions, new_shape)
            new_shape = (ps, int(bs*bp/ps),)
            rewards = jnp.reshape(rewards, new_shape)
            
            # update critic parameters
            critic_state, crit_loss = update_critic(
                    actor_state, critic_state,
                    masks_P, nodes_P, edges_P, i_P, j_P,
                    masks_N, nodes_N, edges_N, i_N, j_N,
                    actions, rewards)
            
            # update actor and target nets parameters
            actor_state, critic_state, actor_loss_value = update_actor(
                    actor_state, critic_state,
                    masks_P, nodes_P, edges_P, i_P, j_P)

            # store rewards
            loss_series.append(crit_loss)
            mean_reward = jnp.mean(rewards)
            reward_series.append(mean_reward)
            print (f'\nCritic loss:{crit_loss}, Reward: {mean_reward}')

    loss_series = list(np.array(loss_series))
    reward_series = list(np.array(reward_series))
    step = list(range(len(loss_series)))

    data = {'steps':step, 'loss':loss_series, 'reward':reward_series}
    
    sb.violinplot(x='labels', y='rewards', data=delta_series)
    plt.savefig('rewards.png')
    sb.violinplot(x='labels', y='rmsds', data=delta_series)
    plt.savefig('rmsds.png')

    g = sb.lineplot(data=data, x='steps', y='loss')
    sb.lineplot(data=data, x='steps', y='reward', color='red', ax=g.axes.twinx())
    plt.savefig('train.png')

    with open('params.pkl', 'wb') as out: pkl.dump(actor_state.params, out)

    ####################################################
    # INDEPENDENT TEST
    ####################################################
#
#    io = PDBIO()
#    pdbp = PDBParser(QUIET=True)
#    test_code = test_code.upper()
#    rpath = f'{path}/data/benchmark5.5/{test_code}_r_b.pdb'
#    lpath = f'{path}/data/benchmark5.5/{test_code}_l_b.pdb'
#
#    
#    out_struc = Structure('test')
#
#    # move test ligands from native
#    all_idxs = jnp.indices((tenvs.number,))[0]
#    key, nkey = jrn.split(key, 2)
#    tenvs.move_from_native(nkey, all_idxs, config['native_dev'])
#
#    for global_step in range(config['test_steps']):
#
#        # get state
#        (mask_ints, edge_ints, i_ints, j_ints) = tenvs.get_state(envs.coord_ligs)
#        
#        # format state
#        (masks, nodes, edges, i, j) = \
#                tenvs.format_input_state(
#                        mask_ints, edge_ints, i_ints, j_ints, all_idxs, 1)
#        # get action
#        actions = partial(a_apply, a_params, None)(masks, nodes, edges, i, j)
#        actions_t = vmap(lambda x: jnp.argmax(x, axis=1))(actions)
#        Ps, p1s, p2s = tenvs.refine_action(tenvs.coord_ligs, actions_t)
#        quats = vmap(quat_from_pivoting)(Ps, p1s, p2s)
#
#        # apply action and get reward
#        tenvs.coord_ligs = tenvs.step(tenvs.coord_ligs, Ps, quats)
#        dmaps = vmap(distances_from_coords)(
#                tenvs.coord_recs[:,:,None,:], tenvs.coord_ligs[:,None,:,:])
#        dmaps = vmap(lambda x,y,z: x*y[:,None]*z[None,:])(
#                dmaps, tenvs.padmask_recs, tenvs.padmask_ligs)
#        reward = tenvs.get_rewards(dmaps)
#
#        # apply action to full atom structure
##        out_models = unfold_entities(out_struc, 'M')
##        out_struc = save_to_model(out_struc, out_models[-1], Ps, quats)
#        
#        print (f'Writing --- Reward: {reward}')
#
#    io.set_structure(out_struc)
   # io.save('test.pdb')
