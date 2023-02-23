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
from networks.updates import *
from replay_buffer import *
from environment import *
from ops import *

from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.Structure import Structure

class TrainState(NamedTuple):
    params: hk.Params
    target_params: hk.Params
    opt_state: optax.OptState

if __name__ == "__main__":    
    
    path = os.getcwd()+'/'+'/'.join(__file__.split('/')[:-2])

    s = time.time()
    with open(path+'/src/config.json') as j: config = json.load(j)

    dataset, code = load_dataset(path+'/data/dataset_features/', size=2)
    test, test_code = load_dataset(path+'/data/dataset_features/', size=1, skip=2)

    io = PDBIO()
    pdbp = PDBParser(QUIET=True)
    test_code = test_code.upper()
    rpath = f'{path}/data/benchmark5.5/{test_code}_r_b.pdb'
    lpath = f'{path}/data/benchmark5.5/{test_code}_l_b.pdb'
    
    out_struc = Structure('test')

    # save structures of starting models
#    rstruc = pdbp.get_structure('', rpath)
#    cm = jnp.array(test[test_code]['init_rt'][0][0])
#    quat = jnp.array(test[test_code]['init_rt'][0][1])
#    out_struc = save_to_model(out_struc, rstruc[0], quat, cm, init=True)

#    lstruc = pdbp.get_structure('', lpath)
#    cm = jnp.array(test[test_code]['init_rt'][1][0])
#    quat = jnp.array(test[test_code]['init_rt'][1][1])
#    out_struc = save_to_model(out_struc, lstruc[0], quat, cm, init=True)

    # devices
    cpus = jax.devices('cpu')
    gpus = jax.devices('gpu')
    #os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

    # seeding
    random.seed(config['random_seed'])
    key = jrn.PRNGKey(config['random_seed'])
    key, env_key, buff_key = jrn.split(key, 3)
    key, act_key, crit_key = jrn.split(key, 3)

    # environment setup
    envs = DockingEnv(dataset, 10, 400, env_key)
    tenvs = DockingEnv(test, 10, 400, env_key)
    
    # replay buffer setup
    r_buffer = ReplayBuffer(config, envs.list)

    # actor/critic models setup
    actor = hk.transform(actor_fw)
    critic = hk.transform(critic_fw)
    a_apply = jax.jit(actor.apply)
    c_apply = jax.jit(critic.apply)
   
    (mask_ints, edge_ints, i_ints, j_ints) = envs.get_state(envs.coord_ligs)

    (mask, nodes, edges, i, j) = \
            envs.format_input_state(mask_ints, edge_ints, i_ints, j_ints)

    # actor parameters initialization
    a_params = actor.init(act_key, mask, nodes, edges, i, j)

    # critic parameters initialization
    action = a_apply(a_params, None, mask, nodes, edges, i, j)
    c_params = critic.init(crit_key, mask, nodes, edges, i, j, action)

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

    print ('Initialization completed! -', time.time()-s)

    # move ligands from native pose
    all_idxs = jnp.indices((envs.number,))[0]
    key, n_key = jrn.split(key, 2)
    envs.move_from_native(n_key, all_idxs, 10, config)
    
    # get state
    (mask_ints, edge_ints, i_ints, j_ints) = envs.get_state(envs.coord_ligs)

    # training iteration
    sg = time.time()
    loss_series = []
    reward_series = []
    rewards_episode = []
    print ('Start training! - Filling buffer...')
    for global_step in range(config['steps']):
        
        if global_step == config['buffer_size']: 
            print (f'Buffer full! - {time.time()-sg}')
            sg = time.time()

        # format state
        (masks, nodes, edges, i, j) = \
                envs.format_input_state(mask_ints, edge_ints, i_ints, j_ints)

        # generate agent action
        actions = partial(a_apply, a_params, None)(masks, nodes, edges, i, j)
        actions_t = vmap(lambda x: jnp.argmax(x, axis=1))(actions)
        Ps, p1s, p2s = envs.refine_action(envs.coord_ligs, actions_t)
        quats = vmap(quat_from_pivoting)(Ps, p1s, p2s)      

        ### add noise to action

        # apply action
        envs.coord_ligs = envs.step(envs.coord_ligs, Ps, quats)
        
        dmaps = vmap(distances_from_coords)(
                envs.coord_recs[:,:,None,:], envs.coord_ligs[:,None,:,:])
        dmaps = vmap(lambda x,y,z: x*y[:,None]*z[None,:])(
                dmaps, envs.padmask_recs, envs.padmask_ligs)
        
        # get reward for last action
        rewards = envs.get_rewards(dmaps)
        rewards_episode.append(rewards)

        # get new state
        (mask_ints_N, edge_ints_N, i_ints_N, j_ints_N) = envs.get_state(envs.coord_ligs)
        mask_ints = jnp.array(mask_ints, dtype=jnp.uint8)
        mask_ints_N = jnp.array(mask_ints_N, dtype=jnp.uint8)
        
        # store experiences in replay buffer
        experience = [mask_ints, edge_ints, i_ints, j_ints,
                mask_ints_N, edge_ints_N, i_ints_N, j_ints_N,
                actions, rewards]

        # fast storage into a reduced size cache on GPU
        r_buffer.cache, r_buffer.cache_actual = \
                r_buffer.add_to_cache(r_buffer.cache, experience, r_buffer.cache_actual)
        
        if r_buffer.cache_actual == r_buffer.cache_size:
            
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

        # update state for next iteration
        mask_ints = mask_ints_N
        edge_ints = edge_ints_N
        i_ints = i_ints_N
        j_ints = j_ints_N

        # check reset conditions
        if illegal_interfaces(dmaps, config, True):
            reset_idxs = get_illegal_idxs(dmaps, config)
            print ('To reset: ', reset_idxs)
            envs.reset(dataset, reset_idxs)

            key, n_key = jrn.split(key, 2)
            envs.move_from_native(n_key, reset_idxs, 10, config)

        # network updates
        if global_step+1 > config['buffer_size'] \
        and (global_step+1) % config['policy_frequency'] == 0:

            key, pkey = jrn.split(key, 2)

            # select a number of protein pairs 
            batch_pair_idx = jrn.choice(
                    pkey, envs.number, shape=(args.batch_pair_num,), replace=False)

            # get corresponding codes
            batch_pair_codes = [envs.list[idx] for idx in batch_pair_idx]
                
            # group buffers for each pair code
            prot_buffers = [r_buffer.buffer[code] for code in batch_pair_codes]

            # extract batch
            batch, r_buffer.key = r_buffer.sample_from_replay(
                    args.batch_size, prot_buffers, r_buffer.key)    
            batch = jax.device_put(batch, device=gpus[0])

            # update critic parameters
            critic_state, crit_loss = update_critic(
                    actor_state, critic_state, batch, batch_pair_idx)

            # update actor and target nets parameters
            actor_state, critic_state, actor_loss_value = update_actor(
                    actor_state, critic_state, batch, batch_pair_idx)

            # store rewards
            mean_reward = jnp.mean(jnp.array(rewards_episode))
            rewards_episode = []
            loss_series.append(crit_loss)
            reward_series.append(mean_reward)
            print (f'Completed episode {global_step+1} - {time.time()-sg}')
            print (f'Critic loss:{crit_loss}, Reward: {mean_reward}')

    loss_series = list(np.array(loss_series))
    reward_series = list(np.array(reward_series))
    step = list(range(len(loss_series)))

    plt.figure()
    data = {'steps':step, 'values':loss_series}
    sb.lineplot(data=data, x='steps', y='values')
    plt.savefig('loss.png')

    plt.figure()
    data = {'steps':step, 'values':reward_series}
    sb.lineplot(data=data, x='steps', y='values')
    plt.savefig('reward.png')

    for global_step in range(args.test_steps):

        # get state
        tenvs.edge_int, tenvs.send_int, tenvs.neigh_int = tenvs.get_state()

        # get action
        actions = vmap(partial(a_apply, a_params, None))(
                tenvs.feat_rec, tenvs.feat_lig, tenvs.mask_rec, tenvs.mask_lig,
                tenvs.edge_rec, tenvs.send_rec, tenvs.neigh_rec,
                tenvs.edge_lig, tenvs.send_lig, tenvs.neigh_lig,
                tenvs.edge_int, tenvs.send_int, tenvs.neigh_int,)

        # apply action
        dmaps, coord_lig = tenvs.step(tenvs.coord_rec, tenvs.coord_lig, actions)

        tenvs.coord_lig = coord_lig
        reward = tenvs.get_rewards(dmaps)

        # apply action to full atom structure
        out_models = unfold_entities(out_struc, 'M')
        out_struc = save_to_model(out_struc, out_models[-1], actions[0,:4], actions[0,4:-1])
        
        print (f'Completed episode {global_step+1} - {time.time()-sg}')
        print (f'Writing --- Reward: {reward}')

    io.set_structure(out_struc)
    io.save('test.pdb')
