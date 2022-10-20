# derived from https://docs.cleanrl.dev/rl-algorithms/ddpg
import os
import time
import glob
import random
import argparse
import functools
import pickle as pkl
from distutils.util import strtobool
from typing import Sequence, Iterable, Mapping, NamedTuple, Tuple

import jax
import optax
import haiku as hk
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
    parser.add_argument("--steps", type=int, default=100,
        help="total number of episodes")
    parser.add_argument("--episode", type=int, default=10,
        help="total number of substeps")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer_size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch_size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--rot_noise", type=float, default=0.01,
        help="the standard dev. of rotation noise")
    parser.add_argument("--tr_noise", type=float, default=4,
        help="the standard dev. of traslation noise")
    parser.add_argument("--learning_starts", type=int, default=25e3,
        help="timestep to start learning")
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
        for lbl in ['clouds', 'orig_clouds', 'masks']:
            data[lbl] = (jnp.array(data[lbl][0]),jnp.array(data[lbl][1]))
        
        data['interface'] = jnp.array(data['interface'])
        
        g1, g2 = data['graphs']

        data['graphs'] = (jraph.GraphsTuple(
                            nodes=jnp.array(g1.nodes), 
                            edges=jnp.array(g1.edges), 
                            senders=jnp.array(g1.senders), 
                            receivers=jnp.array(g1.receivers),
                            n_node=jnp.array(g1.n_node), 
                            n_edge=jnp.array(g1.n_edge), 
                            globals=jnp.array(g1.globals)),
                          jraph.GraphsTuple(
                            nodes=jnp.array(g2.nodes),
                            edges=jnp.array(g2.edges), 
                            senders=jnp.array(g2.senders),
                            receivers=jnp.array(g2.receivers),
                            n_node=jnp.array(g2.n_node),
                            n_edge=jnp.array(g2.n_edge),
                            globals=jnp.array(g2.globals)))

        dataset[code] = data
        if idx == 0: break
    return dataset, code

class TrainState(NamedTuple):
    params: hk.Params
    target_params: hk.Params
    apply_fn: hk.Module
    opt_state: optax.OptState

def actor_fw(g_rec, g_lig, g_int):
    actor = Actor(1, 4)
    return actor(g_rec, g_lig, g_int)

def critic_fw(g_rec, g_lig, g_int, action):
    critic = Critic(1, 4)
    return critic(g_rec, g_lig, g_int, action)


if __name__ == "__main__":
    args = parse_args()

    dataset, code = load_dataset(args.path)
 
    # seeding
    random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, env_key, buff_key = jax.random.split(key, 3)
    key, act_key, crit_key = jax.random.split(key, 3)

    # env setup
    envs = DockingEnv(dataset, env_key)
    envs.reset(dataset)

    r_buffer = ReplayBuffer(
        envs, args.buffer_size, buff_key)

    actor = hk.transform(actor_fw)
    critic = hk.transform(critic_fw)
    a_apply = jax.jit(actor.apply)
    c_apply = jax.jit(critic.apply)
    
    s = time.time()
    # init actor parameters
    obs = envs.get_state()
    unwrapped_obs = jax.tree_util.tree_map(lambda x: x.g, obs)
    print ('Computed first state -', time.time()-s) 
    s = time.time()

    a_params = actor.init(
            act_key, 
            envs.g_rec[code],
            envs.g_lig[code],
            unwrapped_obs[code])
    print ('Initialized actor -', time.time()-s)
    s = time.time()

    # init critic parameters
    action = a_apply(
            a_params, act_key,
            envs.g_rec[code],
            envs.g_lig[code],
            unwrapped_obs[code])
    print ('Computed first action -', time.time()-s)
    s = time.time()

    
    c_params = critic.init(
            crit_key,
            envs.g_rec[code],
            envs.g_lig[code],
            unwrapped_obs[code],
            action)
    print ('Initialized critic -', time.time()-s)

    # duplicate and group actor/critic params
    actor_state = TrainState(
        apply_fn=a_apply,
        params=a_params,
        target_params=a_params,
        opt_state=optax.adam(learning_rate=args.learning_rate))

    critic_state = TrainState(
        apply_fn=c_apply,
        params=c_params,
        target_params=c_params,
        opt_state=optax.adam(learning_rate=args.learning_rate))

    @jax.jit
    def update_critic(
        actor_state, crit_state, 
        states, actions, rewards, next_states):

        next_action = actor.apply(actor_state.target_params, next_state)
        next_q = critic.apply(crit_state.target_params, next_state, next_action)
        y = reward + args.gamma * (next_q)

        def mse_loss(params):
            q = critic.apply(params, state, action)
            return (y - q)**2, q

        (crit_loss, q), grads = jax.value_and_grad(mse_loss, has_aux=True)(crit_state.params)
        crit_state = crit_state.apply_gradients(grads=grads)
        return crit_state, crit_loss, q

    @jax.jit
    def update_actor(actor_state, crit_state, states):

        def actor_loss(params):
            return -critic.apply(crit_state.params, state, actor.apply(params, state)).mean()

        actor_loss, grads = jax.value_and_grad(actor_loss)(actor_state.params)
        
        actor_state = actor_state.apply_gradients(grads=grads)
        
        actor_state = actor_state.replace(
            target_params=optax.incremental_update(
                actor_state.params, 
                actor_state.target_params, 
                args.tau))

        crit_state = crit_state.replace(
            target_params=optax.incremental_update(
                crit_state.params, 
                crit_state.target_params, 
                args.tau))

        return actor_state, crit_state, actor_loss

    print ('Start training!')
    for global_step in range(args.steps):
        
        sg = time.time()
        
        # run actor network
        print (f'New step({global_step})!')
        key_dic1, key_dic2, key_dic3 = {}, {}, {}
        for pdb in envs.list:
            key, nkey1, nkey2, nkey3 = jax.random.split(key, 4)
            key_dic1[pdb], key_dic2[pdb], key_dic3[pdb] = nkey1, nkey2, nkey3

        actions = jax.tree_util.tree_map(
                functools.partial(a_apply, actor_state.params),
                key_dic1, envs.g_rec, envs.g_lig, unwrapped_obs)
        
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
        print ('Computed action -', time.time()-sg) 
        s = time.time()

        # execute action and get reward
        rewards, envs.c_lig, envs.contacts, envs.lookup_table = envs.step(actions)
        print ('Computed reward -', time.time()-s)
        s = time.time()

        # observe next status
        next_obs = envs.get_state()
        print ('Computed state -', time.time()-s)
        s = time.time()

        # store experiences in replay buffer
        experiences = jax.tree_util.tree_map(
                lambda a, b, c, d: Experience(a, b, c, d),
                obs, next_obs, actions, rewards)
        r_buffer.add(experiences)

        # proceed to next step
        obs = next_obs
        unwrapped_obs = jax.tree_util.tree_map(lambda x: x.g, obs)

        # training
        #if global_step > args.learning_starts:
        if False:
            data = r_buffer.sample(args.batch_size)
            qf1_state, qf1_loss_value, qf1_a_values = update_critic(
                actor_state,
                critic_state,
                data.observations.numpy(),
                data.actions.numpy(),
                data.next_observations.numpy(),
                data.rewards.flatten().numpy(),
                data.dones.flatten().numpy(),
            )
            if global_step % args.policy_frequency == 0:
                actor_state, qf1_state, actor_loss_value = update_actor(
                    actor_state,
                    qf1_state,
                    data.observations.numpy(),
                )

        # reset for the end of an episode
        if args.steps % args.episode: 
            envs.reset(dataset)
            obs = envs.get_state()
            unwrapped_obs = jax.tree_util.tree_map(lambda x: x.g, obs)
        print ('Completed episode -', time.time()-sg)
