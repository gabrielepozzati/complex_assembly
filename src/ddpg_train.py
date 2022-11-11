# derived from https://docs.cleanrl.dev/rl-algorithms/ddpg
import os
import time
import glob
import random
import argparse
import functools
import pickle as pkl
import matplotlib.pyplot as plt
from distutils.util import strtobool
from typing import Sequence, Iterable, Mapping, NamedTuple, Tuple

import jax
import optax
import haiku as hk
import replay_buffer
import jax.numpy as jnp
from jax import tree_util
from jraph import GraphsTuple
from networks.critic import Critic
from networks.actor import Actor
from replay_buffer import *
from environment import *
from ops import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, 
        default='/home/pozzati/complex_assembly/data/dataset_features/',
        help="data path")
    parser.add_argument("--seed", type=int, default=42,
        help="seed of the experiment")
    parser.add_argument("--steps", type=int, default=100000,
        help="total number of steps")
    parser.add_argument("--episode", type=int, default=100,
        help="number of steps in an episode")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer_size", type=int, default=int(1e4),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch_size", type=int, default=8,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--rot_noise", type=float, default=0.1,
        help="the standard dev. of rotation noise")
    parser.add_argument("--tr_noise", type=float, default=8,
        help="the standard dev. of traslation noise")
    parser.add_argument("--policy_frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--noise_clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    args = parser.parse_args()
    return args

def load_dataset(path):
    dataset = {}
    for idx, path in enumerate(glob.glob(path+'/*')):
        code = path.split('/')[-1].rstrip('.pkl')
        f = open(path, 'br')
        data = pkl.load(f)

        # restore device array type
        for lbl in ['nodes', 'edges', 'iedges', 'senders', 'isenders', 
                    'receivers', 'ireceivers', 'clouds', 'masks', 'init_rt']:
            data[lbl] = (jnp.array(data[lbl][0]),jnp.array(data[lbl][1]))
        
        dataset[code] = data
        if idx == 10: break

    return dataset, code

class TrainState(NamedTuple):
    params: hk.Params
    target_params: hk.Params
    opt_state: optax.OptState

def actor_fw(n_rec, e_rec, s_rec, r_rec,
             n_lig, e_lig, s_lig, r_lig,
             e_int, s_int, r_int):

    actor = Actor(1, 64)

    return actor(n_rec, e_rec, s_rec, r_rec,
                 n_lig, e_lig, s_lig, r_lig,
                 e_int, s_int, r_int)

def critic_fw(n_rec, e_rec, s_rec, r_rec,
              n_lig, e_lig, s_lig, r_lig,
              e_int, s_int, r_int, action):

    critic = Critic(1, 64)

    return critic(n_rec, e_rec, s_rec, r_rec,
                  n_lig, e_lig, s_lig, r_lig,
                  e_int, s_int, r_int, action)

def adjust_action(action):
    action = jnp.squeeze(action)
    return jnp.concatenate((quat_from_pred(action[:3]), action[3:]), axis=-1)

def map_to_nexts(pa, pc, g_rec, g_lig, g_int_batch):
    act_batch = jax.tree_util.tree_map(
            lambda x: a_apply(pa, act_key, g_rec, g_lig, x.next_state),
            g_int_batch, is_leaf=lambda x: type(x)==Experience)

    act_batch = jax.tree_util.tree_map(lambda x: adjust_action(x),
            act_batch, is_leaf=lambda x: type(x)==Experience)

    return jax.tree_util.tree_map(
            lambda x, y: c_apply(pc, crit_key, g_rec, g_lig, x.prev_state, y),
            g_int_batch, act_batch, is_leaf=lambda x: type(x)==Experience)

def map_to_prevs(pa, pc, g_rec, g_lig, g_int_batch):
    act_batch = jax.tree_util.tree_map(
            lambda x: a_apply(pa, act_key, g_rec, g_lig, x.prev_state),
            g_int_batch, is_leaf=lambda x: type(x)==Experience)
    
    act_batch = jax.tree_util.tree_map(lambda x: adjust_action(x),
            act_batch, is_leaf=lambda x: type(x)==Experience)
    
    return jax.tree_util.tree_map(
            lambda x, y: c_apply(pc, crit_key, g_rec, g_lig, x.prev_state, y),
            g_int_batch, act_batch, is_leaf=lambda x: type(x)==Experience)

def map_to_actions(pc, g_rec, g_lig, g_int_batch):
    act_batch = jax.tree_util.tree_map(lambda x: x.action, g_int_batch,
            is_leaf=lambda x: type(x)==Experience)

    return jax.tree_util.tree_map(
            lambda x, y: c_apply(pc, crit_key, g_rec, g_lig, x.prev_state, y),
            g_int_batch, act_batch, is_leaf=lambda x: type(x)==Experience)

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
    args = parse_args()

    dataset, code = load_dataset(args.path)

    #io = PDBIO()
    #pdbp = PDBParser(QUIET=True)
    #rpath = '/home/pozzati/complex_assembly/data/benchmark5.5/1A2K_r_b.pdb'
    #lpath = '/home/pozzati/complex_assembly/data/benchmark5.5/1A2K_l_b.pdb'
    
    #out_struc = Structure('test')

    # save structures of starting models
    #rstruc = pdbp.get_structure('', rpath)
    #cm = jnp.array(dataset[code]['init_rt'][0][0])
    #quat = jnp.array(dataset[code]['init_rt'][0][1])
    #out_struc = save_to_model(out_struc, rstruc[0], quat, cm, init=True)

    #lstruc = pdbp.get_structure('', lpath)
    #cm = jnp.array(dataset[code]['init_rt'][1][0])
    #quat = jnp.array(dataset[code]['init_rt'][1][1])
    #out_struc = save_to_model(out_struc, lstruc[0], quat, cm, init=True)

    # seeding
    random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, env_key, buff_key = jax.random.split(key, 3)
    key, act_key, crit_key = jax.random.split(key, 3)

    # environment setup
    envs = DockingEnv(dataset, env_key)

    # replay buffer setup
    r_buffer = ReplayBuffer(buff_key, args.buffer_size, 
                            envs.number, 4, 1000)

    # actor/critic models setup
    actor = hk.transform(actor_fw)
    critic = hk.transform(critic_fw)
    a_init = vmap(actor.init)
    c_init = vmap(critic.init)
    a_apply = jax.jit(vmap(actor.apply))
    c_apply = jax.jit(vmap(critic.apply))
    
    # actor parameters initialization
    s = time.time()
    e_int, s_int, r_int = envs.get_state()
    a_params = a_init(act_key, 
            envs.nodes_rec[0], envs.edges_rec[0],
            envs.senders_rec[0], envs.receivers_rec[0],
            envs.nodes_lig[0], envs.edges_lig[0],
            envs.senders_lig[0], envs.receivers_lig[0],
            e_int[0], s_int[0], r_int[0])
    print ('Initialized actor -', time.time()-s)
    s = time.time()

    # critic parameters initialization
    action = a_apply(a_params, act_key,
            envs.nodes_rec[0], envs.edges_rec[0],
            envs.senders_rec[0], envs.receivers_rec[0],
            envs.nodes_lig[0], envs.edges_lig[0],
            envs.senders_lig[0], envs.receivers_lig[0],
            e_int[0], s_int[0], r_int[0])
    
    action = adjust_action(action)

    c_params = critic.init(crit_key,
            envs.nodes_rec[0], envs.edges_rec[0],
            envs.senders_rec[0], envs.receivers_rec[0],
            envs.nodes_lig[0], envs.edges_lig[0],
            envs.senders_lig[0], envs.receivers_lig[0],
            e_int[0], s_int[0], r_int[0], action)
    print ('Initialized critic -', time.time()-s)

    # duplicate and group actor/critic params, init optimizer
    optimiser = optax.adam(learning_rate=args.learning_rate)

    actor_state = TrainState(
        params=a_params,
        target_params=a_params,
        opt_state=optimiser.init(a_params))

    critic_state = TrainState(
        params=c_params,
        target_params=c_params,
        opt_state=optimiser.init(c_params))

    @jax.jit
    def update_critic(actor_state, critic_state, batch):

        s = time.time()
        # map all protein pairs to corresponding batch to compute next-step Q
        next_qs = jax.tree_util.tree_map(
                lambda a, b, c: map_to_nexts(actor_state.target_params, 
                                             critic_state.target_params, b, c, a), 
                batch, envs.graph_rec, envs.graph_lig,
                is_leaf=lambda x: type(x)==list)

        # compute critic loss target across dataset
        ys = jax.tree_util.tree_map(lambda x, n_q: x.reward + args.gamma * (n_q),
                batch, next_qs,
                is_leaf=lambda x: type(x)==Experience)
        print ('Computed targets -', time.time()-s)
        s = time.time()


        def crit_loss_fn(params):
            qs = jax.tree_util.tree_map(
                    lambda a, b, c: map_to_actions(params, b, c, a), 
                    batch, envs.graph_rec, envs.graph_lig,
                    is_leaf=lambda x: type(x)==list)

            losses = jax.tree_util.tree_map(lambda y, q: (y - q)**2, ys, qs)
            losses = jax.tree_util.tree_flatten(losses)[0]
            return jnp.mean(jnp.array(losses))


        crit_loss, grads = jax.value_and_grad(crit_loss_fn)(critic_state.params)
        print ('Computed updates -', time.time()-s)
        s = time.time()

        updates, new_opt_state = optimiser.update(grads, critic_state.opt_state)
        critic_state = critic_state._replace(
                params=optax.apply_updates(critic_state.params, updates),
                opt_state=new_opt_state)
        print ('Updated -', time.time()-s)

        return critic_state, crit_loss

    @jax.jit
    def update_actor(actor_state, critic_state, batch):


        def actor_loss_fn(params):
            losses = jax.tree_util.tree_map(
                    lambda a, b, c: map_to_prevs(params, 
                                                 critic_state.params, b, c, a),
                    batch, envs.graph_rec, envs.graph_lig,
                    is_leaf=lambda x: type(x)==list)

            losses = jax.tree_util.tree_flatten(losses)[0]
            return jnp.mean(-jnp.array(losses))

        s = time.time()
        actor_loss, grads = jax.value_and_grad(actor_loss_fn)(actor_state.params)
        print ('Computed actor updates -', time.time()-s)
        s = time.time()

        updates, new_opt_state = optimiser.update(grads, actor_state.opt_state)
        actor_state = actor_state._replace(
                params=optax.apply_updates(actor_state.params, updates),
                opt_state=new_opt_state)
        print ('Updated -', time.time()-s)
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

    # training
    sg = time.time()
    running_reward = []
    print ('Start training!')
    for global_step in range(args.steps):
        
        # generate PRNG keys
        key_dic1, key_dic2, key_dic3 = {}, {}, {}
        for pdb in envs.list:
            key, nkey1, nkey2, nkey3 = jax.random.split(key, 4)
            key_dic1[pdb], key_dic2[pdb], key_dic3[pdb] = nkey1, nkey2, nkey3

        # run actor
        actions = jax.tree_util.tree_map(
                functools.partial(a_apply, actor_state.params),
                key_dic1, envs.graph_rec, envs.graph_lig, obs,
                is_leaf=lambda x: type(x)==GraphsTuple)

        # add noise to action
        actions = jax.tree_util.tree_map(
                lambda a, x, y, z: jnp.concatenate(
                    (jnp.squeeze(a[:,:3] + jax.random.normal(x, shape=(3,))*args.rot_noise),
                     jnp.squeeze(a[:,3:-1] + jax.random.normal(y, shape=(3,))*args.tr_noise),
                     a[:,-1] + jax.random.normal(z, shape=(1,))*args.rot_noise),
                    axis=-1),
                actions, key_dic1, key_dic2, key_dic3)

        actions = jax.tree_util.tree_map(
                lambda x: jnp.concatenate((quat_from_pred(x[:3]), x[3:]), axis=-1), actions)
        #print ('Computed action -', time.time()-sg) 
        #s = time.time()
        
        # execute action and get reward
        rewards, envs.cloud_lig, envs.contacts, envs.visit_lookup = envs.step(envs.cloud_lig, actions)
        envs.scloud_lig = envs.mask_filter(envs.cloud_lig, envs.smask_lig)
        envs.bcloud_lig = envs.mask_filter(envs.cloud_lig, envs.bmask_lig)
        #print ('Computed reward -', time.time()-s)
        #s = time.time()

        # observe next status
        next_obs = envs.get_state()
        #print ('Computed state -', time.time()-s)
        #s = time.time()

        # store experiences in replay buffer
        experiences = jax.tree_util.tree_map(
                lambda a, b, c, d: Experience(prev_state=a, next_state=b, action=c, reward=d),
                obs, next_obs, actions, rewards,
                is_leaf=lambda x: type(x)==GraphsTuple)
        r_buffer = add_to_replay(experiences, r_buffer)

        # update state for next iteration
        obs = next_obs

        # network updates
        if global_step+1 > args.buffer_size:
            
            # sample batch from replay buffer
            gpus = jax.devices('gpu')
            batch, r_buffer = sample_from_replay(args.batch_size, r_buffer)
            batch = jax.device_put(batch, device=gpus[0])

            # update critic parameters
            #critic_state, crit_loss = update_critic(actor_state, critic_state, batch)

            # update actor and target nets parameters
            #actor_state, critic_state, actor_loss_value = update_actor(
            #    actor_state, critic_state, batch)

            # store rewards
            if len(running_reward) == args.episode: running_reward[1:].append(rewards[code])
            else: running_reward.append(rewards[code])

        # apply action to full atom structure
        if global_step >= 9900: 
            out_models = unfold_entities(out_struc, 'M')
            out_struc = save_to_model(out_struc, out_models[-1], actions[code][:4], actions[code][4:-1])

        # reset for the end of an episode
        if global_step+1 % args.episode == 0: 
            envs.reset(dataset)
            obs = envs.get_state()
            if global_step+1 > args.buffer_size:
                mean_reward = jnp.mean(jnp.array(running_reward))
                print (f'Completed episode {global_step+1} - {time.time()-sg}')
                print (f'Actor loss:{actor_loss_value}, Actor reward: {mean_reward}')
    
    io.set_structure(out_struc)
    io.save('test.pdb')
