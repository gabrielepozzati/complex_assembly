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
from networks.equivariant_nets import Actor
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, 
        default='/home/pozzati/complex_assembly/',
        help="data path")
    parser.add_argument("--seed", type=int, default=42,
        help="seed of the experiment")
    parser.add_argument("--steps", type=int, default=20000,
        help="total number of steps")
    parser.add_argument("--episode_steps", type=int, default=100,
        help="total number of steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer_size", type=int, default=1000,
        help="the replay memory buffer size")
    parser.add_argument("--test_steps", type=int, default=10,
        help="the size of output PDB in steps")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch_size", type=int, default=16,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--batch_pair_num", type=int, default=1,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--rot_noise", type=float, default=0.001,
        help="the standard dev. of rotation noise")
    parser.add_argument("--tr_noise", type=float, default=1,
        help="the standard dev. of traslation noise")
    parser.add_argument("--policy_frequency", type=int, default=100,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--noise_clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    args = parser.parse_args()
    
    s = time.time()
    with open(args.path+'/src/config.json') as j:
        config = json.load(j)

    dataset, code = load_dataset(args.path+'data/dataset_features/', size=2)
    test, test_code = load_dataset(args.path+'data/dataset_features/', size=1, skip=2)

    io = PDBIO()
    pdbp = PDBParser(QUIET=True)
    test_code = test_code.upper()
    rpath = f'/home/pozzati/complex_assembly/data/benchmark5.5/{test_code}_r_b.pdb'
    lpath = f'/home/pozzati/complex_assembly/data/benchmark5.5/{test_code}_l_b.pdb'
    
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

    # seeding
    random.seed(args.seed)
    key = jrn.PRNGKey(args.seed)
    key, env_key, buff_key = jrn.split(key, 3)
    key, act_key, crit_key = jrn.split(key, 3)

    # environment setup
    envs = DockingEnv(dataset, 10, 400, env_key)
    tenvs = DockingEnv(test, 10, 400, env_key)
    #print (envs_test.list)
    #print (envs_test.e_rec.shape, envs_test.s_rec.shape, envs_test.r_rec.shape, envs_test.n_rec.shape,
    #       envs_test.e_lig.shape, envs_test.s_lig.shape, envs_test.r_lig.shape, envs_test.n_lig.shape,
    #       envs_test.e_int.shape, envs_test.s_int.shape, envs_test.r_int.shape)

    # replay buffer setup
    r_buffer = ReplayBuffer(buff_key, args.buffer_size,
                            envs.list, 10, 400, cpus[0])

    def actor_fw(f_rec, f_lig, m_rec, m_lig,
                 e_rec, s_rec, n_rec,
                 e_lig, s_lig, n_lig,
                 e_int, s_int, n_int):

        actor = Actor('actor', 5, True, config)

        return actor(f_rec, f_lig, m_rec, m_lig,
                     e_rec, s_rec, r_rec,
                     e_lig, s_lig, r_lig,
                     e_int, s_int, n_int)

    def critic_fw(f_rec, f_lig, m_rec, m_lig,
                  e_rec, s_rec, n_rec,
                  e_lig, s_lig, n_lig,
                  e_int, s_int, n_int, action):

        critic = Critic('critic', 5, True, config)

        return critic(f_rec, f_lig, m_rec, m_lig,
                      e_rec, s_rec, n_rec,
                      e_lig, s_lig, n_lig,
                      e_int, s_int, n_int, action)

#    # actor/critic models setup
#    actor = hk.transform(actor_fw)
#    critic = hk.transform(critic_fw)
#    a_apply = jax.jit(actor.apply)
#    c_apply = jax.jit(critic.apply)
#    
#    # actor parameters initialization
#    a_params = actor.init(act_key, 
#            envs.feat_rec[0], envs.feat_lig[0], envs._rec[0], envs.m_lig[0],
#            envs.e_rec[0], envs.s_rec[0], envs.n_rec[0],
#            envs.e_lig[0], envs.s_lig[0], envs.n_lig[0],
#            envs.e_int[0], envs.s_int[0], envs.n_int[0])
#
#    # critic parameters initialization
#    action = a_apply(a_params, None,
#            envs.f_rec[0], envs.f_lig[0], envs.m_rec[0], envs.m_lig[0],
#            envs.e_rec[0], envs.s_rec[0], envs.n_rec[0],
#            envs.e_lig[0], envs.s_lig[0], envs.n_lig[0],
#            envs.e_int[0], envs.s_int[0], envs.n_int[0])
#    
#    action = adjust_action(action[None,:])
#
#    c_params = critic.init(crit_key,
#            envs.f_rec[0], envs.f_lig[0], envs.m_rec[0], envs.m_lig[0],
#            envs.e_rec[0], envs.s_rec[0], envs.n_rec[0],
#            envs.e_lig[0], envs.s_lig[0], envs.n_lig[0],
#            envs.e_int[0], envs.s_int[0], envs.n_int[0],
#            action)
#
#    # duplicate and group actor/critic params, init optimizer
#    a_optimiser = optax.adam(learning_rate=args.learning_rate)
#    c_optimiser = optax.adam(learning_rate=args.learning_rate*10)
#
#    actor_state = TrainState(
#        params=a_params,
#        target_params=a_params,
#        opt_state=a_optimiser.init(a_params))
#
#    critic_state = TrainState(
#        params=c_params,
#       jnp.where( target_params=c_params,
#        opt_state=c_optimiser.init(c_params))

    print ('Initialization completed! -', time.time()-s)

    @jax.jit
    def update_critic(actor_state, critic_state, batch, idxs):
        s = time.time()

        #get batch sizes
        bs = (batch['next_edges'].shape[0],)
        fs = envs.feat_rec.shape[-2:]
        ms = envs.mask_rec.shape[-1:]
        es = envs.edge_rec.shape[-2:]
        ss = envs.send_rec.shape[-1:]

        feat_rec_batch = jnp.broadcast_to(envs.feat_rec[idxs][None,:,:], shape=bs+fs)
        feat_lig_batch = jnp.broadcast_to(envs.feat_lig[idxs][None,:,:], shape=bs+fs)
        mask_rec_batch = jnp.broadcast_to(envs.mask_rec[idxs][None,:], shape=bs+ms)
        mask_lig_batch = jnp.broadcast_to(envs.mask_lig[idxs][None,:], shape=bs+ms)
        edge_rec_batch = jnp.broadcast_to(envs.edge_rec[idxs][None,:,:], shape=bs+es)
        send_rec_batch = jnp.broadcast_to(envs.send_rec[idxs][None,:], shape=bs+ss)
        neigh_rec_batch = jnp.broadcast_to(envs.neigh_rec[idxs][None,:], shape=bs+ss)
        edge_lig_batch = jnp.broadcast_to(envs.edge_lig[idxs][None,:,:], shape=bs+es)
        send_lig_batch = jnp.broadcast_to(envs.send_lig[idxs][None,:], shape=bs+ss)
        neigh_lig_batch = jnp.broadcast_to(envs.neigh_lig[idxs][None,:], shape=bs+ss)

        # map all protein pairs to corresponding batch to compute next-step Q
        a_apply = partial(a_apply, actor_state.target_params, None)
        actions = vmap(a_apply)(
                feat_rec_batch, feat_lig_batch, mask_rec_batch, mask_lig_batch,
                edge_rec_batch, send_rec_batch, neigh_rec_batch,
                edge_lig_batch, send_lig_batch, neigh_lig_batch,
                batch['next_edges'], batch['next_senders'], batch['next_receivers'])

        c_apply = partial(c_apply, critic_state.target_params, None)
        next_q = vmap(c_apply)(
                feat_rec_batch, feat_lig_batch, mask_rec_batch, mask_lig_batch,
                edge_rec_batch, send_rec_batch, neigh_rec_batch,
                edge_lig_batch, send_lig_batch, neigh_lig_batch,
                batch['next_edges'], batch['next_senders'], batch['next_receivers'], actions)

        # compute critic loss targets across batch
        y = batch['rewards'] + (args.gamma*next_q)

        def crit_loss_fn(params):
            c_apply = partial(c_apply, params, None)
            q = vmap(c_apply)(
                    feat_rec_batch, feat_lig_batch, mask_rec_batch, mask_lig_batch,
                    edge_rec_batch, send_rec_batch, neigh_rec_batch,
                    edge_lig_batch, send_lig_batch, neigh_lig_batch,
                    batch['prev_edges'], batch['prev_senders'], 
                    batch['prev_receivers'], batch['actions'])

            return jnp.mean((y - q)**2)

        crit_loss, grads = jax.value_and_grad(crit_loss_fn)(critic_state.params)

        updates, new_opt_state = c_optimiser.update(grads, critic_state.opt_state)
        critic_state = critic_state._replace(
                params=optax.apply_updates(critic_state.params, updates),
                opt_state=new_opt_state)

        print ('Updated critic -', time.time()-s)
        return critic_state, crit_loss

    @jax.jit
    def update_actor(actor_state, critic_state, batch, idxs):

        #get batch sizes
        bs = (batch['next_edges'].shape[0],)
        fs = envs.feat_rec.shape[-2:]
        ms = envs.mask_rec.shape[-1:]
        es = envs.edge_rec.shape[-2:]
        ss = envs.send_rec.shape[-1:]

        feat_rec_batch = jnp.broadcast_to(envs.feat_rec[idxs][None,:,:], shape=bs+fs)
        feat_lig_batch = jnp.broadcast_to(envs.feat_lig[idxs][None,:,:], shape=bs+fs)
        mask_rec_batch = jnp.broadcast_to(envs.mask_rec[idxs][None,:], shape=bs+ms)
        mask_lig_batch = jnp.broadcast_to(envs.mask_lig[idxs][None,:], shape=bs+ms)
        edge_rec_batch = jnp.broadcast_to(envs.edge_rec[idxs][None,:,:], shape=bs+es)
        send_rec_batch = jnp.broadcast_to(envs.send_rec[idxs][None,:], shape=bs+ss)
        neigh_rec_batch = jnp.broadcast_to(envs.neigh_rec[idxs][None,:], shape=bs+ss)
        edge_lig_batch = jnp.broadcast_to(envs.edge_lig[idxs][None,:,:], shape=bs+es)
        send_lig_batch = jnp.broadcast_to(envs.send_lig[idxs][None,:], shape=bs+ss)
        neigh_lig_batch = jnp.broadcast_to(envs.neigh_lig[idxs][None,:], shape=bs+ss)

        def actor_loss_fn(params):
            a_apply = partial(a_apply, params, None)
            actions = vmap(a_apply_params)(
                    feat_rec_batch, feat_lig_batch, mask_rec_batch, mask_lig_batch,
                    edge_rec_batch, send_rec_batch, neigh_rec_batch,
                    edge_lig_batch, send_lig_batch, neigh_lig_batch,
                    batch['prev_edges'], batch['prev_senders'], batch['prev_receivers'])

            c_apply = partial(c_apply, critic_state.params, None)
            q = vmap(c_apply)(
                    feat_rec_batch, feat_lig_batch, mask_rec_batch, mask_lig_batch,
                    edge_rec_batch, send_rec_batch, neigh_rec_batch,
                    edge_lig_batch, send_lig_batch, neigh_lig_batch,
                    batch['prev_edges'], batch['prev_senders'], batch['prev_receivers'], actions)

            return jnp.mean(-q)

        s = time.time()
        actor_loss, grads = jax.value_and_grad(actor_loss_fn)(actor_state.params)

        updates, new_opt_state = a_optimiser.update(grads, actor_state.opt_state)
        actor_state = actor_state._replace(
                params=optax.apply_updates(actor_state.params, updates),
                opt_state=new_opt_state)
        print ('Updated actor-', time.time()-s)
        s = time.time()

        
        actor_state = actor_state._replace(
            target_params=optax.incremental_update(
                actor_state.params, 
                actor_state.target_params, 
                args.tau))

        critic_state = critic_state._replace(
            target_params=optax.incremental_update(
                critic_state.params, 
                critic_state.target_params, 
                args.tau))

        print ('Updated target nets -', time.time()-s)
        return actor_state, critic_state, actor_loss

    # training iteration
    sg = time.time()
    print ('Start training!')
    print ('Filling buffer...')
    loss_series = []
    reward_series = []
    rewards_episode = []
    for global_step in range(args.steps):
        
        if global_step == args.buffer_size: 
            print (f'Buffer full! - {time.time()-sg}')
            sg = time.time()

        # get state
        (envs.edge_int, envs.send_int, envs.neigh_int,
         envs.intmask_rec, envs.intmask_lig,
         envs.rimmask_rec, envs.rimmask_lig) = envs.get_state()

        # generate random action
        key, keyn = jrn.split(key, 2)
        actions = envs.get_random_action(keyn)

        # generate agent action
#        actions = vmap(partial(a_apply, a_params, None))(
#                envs.feat_rec, envs.feat_lig, envs.mask_rec, envs.mask_lig, 
#                envs.edge_rec, envs.send_rec, envs.neigh_rec,
#                envs.edge_lig, envs.send_lig, envs.neigh_lig,
#                envs.edge_int, envs.send_int, envs.neigh_int,)
        
        # add noise to action

        # apply action
        dmaps, coord_lig = envs.step(envs.coord_rec, envs.coord_lig, actions)
        
        # get new state
        (edge_int, send_int, neigh_int,
         intmask_rec, intmask_lig, rimmask_rec, rimmask_lig) = envs.get_state()

        # get reward for last action
        contacts, clashes, dist = envs.get_rewards(dmaps)
        print (f'Contacts: {contacts}, Clashes: {clashes}, Distances: {dist}')
        rewards = vmap(lambda x, y, z: x+(y-z), contacts, clashes, dist)
        rewards_episode.append(rewards)

        # store experiences in replay buffer
        r_buffer.buffer, r_buffer.actual_size = r_buffer.add_to_replay(
            r_buffer.buffer,
            {'prev_edges':envs.edge_lig, 'next_edges':edge_lig,
             'prev_senders':envs.send_lig, 'next_senders':send_lig,
             'prev_receivers':envs.neigh_lig, 'next_receivers':neigh_lig,
             'actions':actions, 'rewards':rewards})

        # update state for next iteration
        envs.coord_lig = coord_lig
        envs.edge_int = edge_int
        envs.send_int = send_int
        envs.neigh_int = neigh_int

        # check reset conditions
        chains_distances = jnp.where(dmaps==0, 1e9, dmaps)
        clashes = jnp.sum(jnp.where(chains_distances<4, 1, 0))
        if jnp.all(chains_distances>12) or clashes > 20:
            reset_idxs = jnp.ravel(jnp.indices((envs.number,)))
            envs.reset(dataset, reset_idxs)
            print ('Reset!')

        # network updates
        #if global_step+1 > args.buffer_size \
        #and (global_step+1) % args.policy_frequency == 0:
        if False:    
            gpus = jax.devices('gpu')
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
