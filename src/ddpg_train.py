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



def parse_args():
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
    return args

def load_dataset(path, size=None, skip=0):
    count = 0
    dataset = {}
    for idx, path in enumerate(glob.glob(path+'/*')):
        if idx < skip: continue
        print (path)
        code = path.split('/')[-1].rstrip('.pkl')
        f = open(path, 'br')
        data = pkl.load(f)

        # restore device array type
        for lbl in ['coord_N', 'coord_C', 'coord_CA', 'nodes', 'masks']:
            data[lbl] = (jnp.array(data[lbl][0]),jnp.array(data[lbl][1]))

        dataset[code] = data
        count += 1
        if count == size: break

    return dataset, code

class TrainState(NamedTuple):
    params: hk.Params
    target_params: hk.Params
    opt_state: optax.OptState

def adjust_action(action):
    return jnp.concatenate((quat_from_pred(action[:,:3]), action[:,3:]), axis=-1)

def save_to_model(out_struc, ref_model, quat, tr, init=False):
    atoms = unfold_entities(ref_model, 'A')
    cloud = jnp.array([atom.get_coord() for atom in atoms])

    if init:
        cloud = quat_rotation(cloud-tr, quat)
    else:
        cloud_cm = jnp.mean(cloud, axis=0)
        cloud = quat_rotation(cloud-cloud_cm, quat)+cloud_cm
        cloud += tr

    model_num = len(unfold_entities(out_struc, 'M'))
    model = copy.deepcopy(ref_model)
    model.detach_parent()
    model.id = model_num+1
    model.set_parent(out_struc)
    out_struc.add(model)

    atoms = unfold_entities(model, 'A')
    for xyz, atom in zip(cloud, atoms): atom.set_coord(xyz)
    
    return out_struc

if __name__ == "__main__":
    s = time.time()
    args = parse_args()

    with open(args.path+'/src/config.json') as j:
        config = json.load(j)

    dataset, code = load_dataset(args.path+'data/dataset_features/', size=1)
    test, test_code = load_dataset(args.path+'data/dataset_features/', size=1)

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
    envs_test = DockingEnv(test, 10, 400, env_key)
    #print (envs_test.list)
    #print (envs_test.e_rec.shape, envs_test.s_rec.shape, envs_test.r_rec.shape, envs_test.n_rec.shape,
    #       envs_test.e_lig.shape, envs_test.s_lig.shape, envs_test.r_lig.shape, envs_test.n_lig.shape,
    #       envs_test.e_int.shape, envs_test.s_int.shape, envs_test.r_int.shape)

    # replay buffer setup
    r_buffer = ReplayBuffer(buff_key, args.buffer_size,
                            envs.list, 10, 400, cpus[0])

    def actor_fw(c_rec, f_rec, e_rec, s_rec, r_rec, m_rec,
                 c_lig, f_lig, e_lig, s_lig, r_lig, m_lig):

        actor = Actor('actor', 5, True, config)

        return actor(c_rec, f_rec, e_rec, s_rec, r_rec, m_rec,
                     c_lig, f_lig, e_lig, s_lig, r_lig, m_lig)

    def critic_fw(c_rec, f_rec, e_rec, s_rec, r_rec, m_rec,
                  c_lig, f_lig, e_lig, s_lig, r_lig, m_lig, action):

        critic = Critic('critic', 5, True, config)

        return critic(c_rec, f_rec, e_rec, s_rec, r_rec, m_rec,
                      c_lig, f_lig, e_lig, s_lig, r_lig, m_lig, action)

    # actor/critic models setup
    actor = hk.transform(actor_fw)
    critic = hk.transform(critic_fw)
    a_apply = jax.jit(actor.apply)
    c_apply = jax.jit(critic.apply)
    
    # actor parameters initialization
    a_params = actor.init(act_key, 
            envs.c_rec_CA[0], envs.f_rec[0], envs.edges_rec[0],
            envs.nodes_rec[0], envs.neighs_rec[0], envs.l_rec[0],
            envs.c_lig_CA[0], envs.f_lig[0], envs.edges_lig[0],
            envs.nodes_lig[0], envs.neighs_lig[0], envs.l_lig[0])

    # critic parameters initialization
    action = a_apply(a_params, None,
            envs.c_rec_CA[0], envs.f_rec[0], envs.edges_rec[0],
            envs.nodes_rec[0], envs.neighs_rec[0], envs.l_rec[0],
            envs.c_lig_CA[0], envs.f_lig[0], envs.edges_lig[0],
            envs.nodes_lig[0], envs.neighs_lig[0], envs.l_lig[0])
    
    action = adjust_action(action[None,:])

    c_params = critic.init(crit_key,
            envs.c_rec_CA[0], envs.f_rec[0], envs.e_rec[0], 
            envs.s_rec[0], envs.r_rec[0], envs.m_rec[0],
            envs.c_lig_CA[0], envs.f_lig[0], envs.e_lig[0], 
            envs.s_lig[0], envs.r_lig[0], envs.m_rec[0],
            action)

    # duplicate and group actor/critic params, init optimizer
    a_optimiser = optax.adam(learning_rate=args.learning_rate)
    c_optimiser = optax.adam(learning_rate=args.learning_rate*10)

    actor_state = TrainState(
        params=a_params,
        target_params=a_params,
        opt_state=a_optimiser.init(a_params))

    critic_state = TrainState(
        params=c_params,
        target_params=c_params,
        opt_state=c_optimiser.init(c_params))

    print ('Initialization completed! -', time.time()-s)

    @jax.jit
    def update_critic(actor_state, critic_state, batch, idxs):
        s = time.time()

        # map all protein pairs to corresponding batch to compute next-step Q
        a_apply_params = partial(a_apply, actor_state.target_params, None)
        a_apply_dataset = partial(vmap(a_apply_params),
                envs.e_rec[idxs], envs.s_rec[idxs], envs.r_rec[idxs], envs.n_rec[idxs],
                envs.e_lig[idxs], envs.s_lig[idxs], envs.r_lig[idxs])

        actions = vmap(a_apply_dataset)(batch['next_nodes'], batch['next_edges'],
                batch['next_senders'], batch['next_receivers'])

        actions = vmap(adjust_action)(actions)

        c_apply_params = partial(c_apply, critic_state.target_params, None)
        c_apply_dataset = partial(vmap(c_apply_params),
                envs.e_rec[idxs], envs.s_rec[idxs], envs.r_rec[idxs], envs.n_rec[idxs],
                envs.e_lig[idxs], envs.s_lig[idxs], envs.r_lig[idxs])
        next_q = vmap(c_apply_dataset)(batch['next_nodes'], batch['next_edges'], 
                batch['next_senders'], batch['next_receivers'], actions)

        # compute critic loss targets across batch
        y = batch['rewards'] + (args.gamma*next_q)

        def crit_loss_fn(params):
            c_apply_params = partial(c_apply, params, None)
            c_apply_dataset = partial(vmap(c_apply_params),
                    envs.e_rec[idxs], envs.s_rec[idxs], envs.r_rec[idxs], envs.n_rec[idxs],
                    envs.e_lig[idxs], envs.s_lig[idxs], envs.r_lig[idxs])
            q = vmap(c_apply_dataset)(batch['prev_nodes'], batch['prev_edges'], 
                    batch['prev_senders'], batch['prev_receivers'], batch['actions'])

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

        def actor_loss_fn(params):
            a_apply_params = partial(a_apply, params, None)
            a_apply_dataset = partial(vmap(a_apply_params),
                envs.e_rec[idxs], envs.s_rec[idxs], envs.r_rec[idxs], envs.n_rec[idxs],
                envs.e_lig[idxs], envs.s_lig[idxs], envs.r_lig[idxs])
            actions = vmap(a_apply_dataset)(batch['prev_nodes'], batch['prev_edges'], 
                    batch['prev_senders'], batch['prev_receivers'])

            actions = vmap(adjust_action)(actions)

            c_apply_params = partial(c_apply, critic_state.params, None)
            c_apply_dataset = partial(vmap(c_apply_params),
                    envs.e_rec[idxs], envs.s_rec[idxs], envs.r_rec[idxs], envs.n_rec[idxs],
                    envs.e_lig[idxs], envs.s_lig[idxs], envs.r_lig[idxs])
            q = vmap(c_apply_dataset)(batch['prev_nodes'], batch['prev_edges'], 
                    batch['prev_senders'], batch['prev_receivers'], actions)

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

        # generate PRNG keys
        key, keys1, keys2, keys3 = jrn.split(key, 4)
        keys1 = jnp.stack(jrn.split(key, envs.number))
        keys2 = jnp.stack(jrn.split(key, envs.number))
        keys3 = jnp.stack(jrn.split(key, envs.number))

        # run actor
        edges, nodes, neighs = envs.get_edges(
                c_lig_N, c_lig_N, c_lig_C, c_lig_C, c_lig_CA, c_lig_CA, 10)

        actions = vmap(partial(a_apply, a_params, None))(
                envs.c_rec, envs.f_rec, envs.e_rec, envs.s_rec, envs.r_rec, envs.m_rec,
                envs.c_lig, envs.f_lig, envs.e_lig, envs.s_lig, envs.r_lig, envs.m_lig)

        # add noise to action
        noiseless = actions
        actions = vmap(lambda x, y, z: jnp.concatenate(
                (jnp.squeeze(x[:9] + jrn.normal(y, shape=(9,))*args.rot_noise),
                 jnp.squeeze(x[-3:] + jrn.normal(z, shape=(3,))*args.tr_noise)),
                axis=-1))(actions, keys1, keys2)

        #actions = jnp.concatenate((quat_from_pred(actions[:,:3]), actions[:,3:]), axis=-1)
        
        # get new state after executing action
        dmaps, c_lig_N, c_lig_C, c_lig_CA = envs.step(
                envs.c_lig_N, envs.c_lig_C, envs.c_lig_CA, actions)
        
        edges, nodes, neighs = envs.get_edges(
                c_lig_N, c_lig_N, c_lig_C, c_lig_C, c_lig_CA, c_lig_CA, 10)

        # get reward for last action
        rewards = envs.get_rewards(dmaps)
        rewards_episode.append(rewards)

        # store experiences in replay buffer
        r_buffer.buffer, r_buffer.actual_size = r_buffer.add_to_replay(
            r_buffer.buffer,
            {'prev_coord':envs.c_lig_CA, 'next_coord':c_lig_CA,
             #'prev_edges':envs.e_lig, 'next_edges':e_lig_next,
             #'prev_senders':envs.s_lig, 'next_senders':s_lig_next,
             #'prev_receivers':envs.r_lig, 'next_receivers':r_lig_next,
             'actions':actions, 'rewards':rewards})

        # update state for next iteration
        envs.c_lig = c_lig_next
        envs.e_int = e_int_next
        envs.s_int = s_int_next
        envs.r_int = r_int_next
        envs.n_lig = n_lig_next

        # check reset conditions
        chains_distances = jnp.where(dmaps==0, 1e9, dmaps)
        clashes = jnp.sum(jnp.where(chains_distances<4, 1, 0))

        if (global_step+1)%args.episode_steps == 0 \
        or (global_step+1) == args.buffer_size \
        or jnp.all(chains_distances>12) \
        or clashes > 50:
            reset_idxs = jnp.ravel(jnp.indices((envs.number,)))
            envs.reset(dataset, reset_idxs)
            # if check hard limit for chain distances
            #chains_distances = jnp.min(jnp.where(dmaps==0, 1e9, dmaps), axis=0)
            #reset_idxs = jnp.ravel(jnp.argwhere(chains_distances>16))
            # update single pairs state for next iteration
            #if reset_idxs.shape[0] > 0: envs.reset(dataset, reset_idxs)

        # network updates
        if global_step+1 > args.buffer_size \
        and (global_step+1) % args.policy_frequency == 0:
            
            # sample batch from replay buffer
            gpus = jax.devices('gpu')
            
            key, pkey = jrn.split(key, 2)
            batch_idx_order = jrn.choice(
                    pkey, envs.number, shape=(envs.number,), replace=False)
            batch_idx_order = list([idx for idx in batch_idx_order])

            while len(batch_idx_order) >= args.batch_pair_num:
                prot_idxs = ()
                for _ in range(args.batch_pair_num):
                    prot_idxs += (int(batch_idx_order.pop(0)),)

                prot_batch_list = [envs.list[idx] for idx in prot_idxs]
                prot_buffers = [r_buffer.buffer[code] for code in prot_batch_list]

                batch, r_buffer.key = r_buffer.sample_from_replay(
                        args.batch_size, prot_buffers, r_buffer.key)
                
                batch = jax.device_put(batch, device=gpus[0])
                prot_idxs = jnp.array(prot_idxs)

                # update critic parameters
                critic_state, crit_loss = update_critic(
                        actor_state, critic_state, batch, prot_idxs)

                # update actor and target nets parameters
                actor_state, critic_state, actor_loss_value = update_actor(
                        actor_state, critic_state, batch, prot_idxs)

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
        # run actor
        actions = a_apply(a_params, None,
                envs_test.e_rec[0], envs_test.s_rec[0], envs_test.r_rec[0], envs_test.n_rec[0],
                envs_test.e_lig[0], envs_test.s_lig[0], envs_test.r_lig[0], envs_test.n_lig[0],
                envs_test.e_int[0], envs_test.s_int[0], envs_test.r_int[0])
       
        actions = jnp.concatenate((quat_from_pred(actions[None,:3]), actions[None,3:]), axis=-1)

        dmaps, c_lig_next, \
        e_int_next, s_int_next, r_int_next = envs_test.step(envs_test.c_lig, actions)
        n_lig_next = jnp.concatenate(
                [envs_test.n_lig[:,:,:-3], c_lig_next], axis=-1)

        envs_test.c_lig = c_lig_next
        envs_test.e_int = e_int_next
        envs_test.s_int = s_int_next
        envs_test.r_int = r_int_next
        envs_test.n_lig = n_lig_next

        # apply action to full atom structure
        out_models = unfold_entities(out_struc, 'M')
        out_struc = save_to_model(out_struc, out_models[-1], actions[0,:4], actions[0,4:-1])
        
        reward = envs_test.get_rewards(dmaps)
        print (f'Completed episode {global_step+1} - {time.time()-sg}')
        print (f'Writing --- Reward: {reward}')

    io.set_structure(out_struc)
    io.save('test.pdb')
