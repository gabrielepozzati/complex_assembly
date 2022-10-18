# derived from https://docs.cleanrl.dev/rl-algorithms/ddpg
import os
import time
import glob
import random
import argparse
import pickle as pkl
from distutils.util import strtobool
from typing import Sequence

import jax
import flax
import optax
import haiku as hk
import jax.numpy as jnp
import flax.linen as nn
from jax import tree_util
from flax.training.train_state import TrainState
from networks.critic import Critic
from networks.actor import Actor
from replay_buffer import *
from environment import *
from ops import *

tree_util.register_pytree_node(
        Experience,
        Experience._tree_flatten,
        Experience._tree_unflatten)

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
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--rot-noise", type=float, default=0.01,
        help="the standard dev. of rotation noise")
    parser.add_argument("--tr-noise", type=float, default=4,
        help="the standard dev. of traslation noise")
    parser.add_argument("--learning-starts", type=int, default=25e3,
        help="timestep to start learning")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    args = parser.parse_args()
    return args


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


if __name__ == "__main__":
    args = parse_args()

    dataset = {}
    for idx, path in enumerate(glob.glob(args.path+'/*')):
        code = path.split('/')[-1].rstrip('.pkl')
        with open(path, 'br') as f: dataset[code] = pkl.load(f)
        if idx == 9: break

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

    actor = hk.transform(Actor)
    critic = hk.transform(Critic)
    actor_apply = jax.jit(actor.apply)
    critc_apply = jax.jit(critic.apply)
    
    # init actor parameters
    obs = envs.get_state()
    actor_params = actor.init(
            act_key, 
            envs.g_rec[code][0], 
            envs.g_lig[code][0], 
            obs[code])

    # init critic parameters
    action = actor_apply(
            actor_params,
            envs.g_rec[code][0],
            envs.g_lig[code][0],
            obs[code])

    critic_params = critic.init(
            crit_key,
            envs.g_rec[code][0],
            envs.g_lig[code][0],
            obs[code],
            action)

    # duplicate and group actor/critic params
    actor_state = TrainState.create(
        apply_fn=actor_apply,
        params=actor_params,
        target_params=actor_params,
        tx=optax.adam(learning_rate=args.learning_rate))

    critic_state = TrainState.create(
        apply_fn=critic_apply,
        params=critic_params,
        target_params=critic_params,
        tx=optax.adam(learning_rate=args.learning_rate))

    @jax.jit
    def update_critic(
        actor_state: TrainState,
        crit_state: TrainState,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray):

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
    def update_actor(
        actor_state: TrainState,
        crit_state: TrainState,
        states: np.ndarray):

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

    obs = envs.get_state()    
    start_time = time.time()
    for global_step in range(args.steps):
        
        # run actor network
        actions = jax.tree_util.tree_map(
                functools.partial(actor_apply, actor_state.params),
                envs.g_rec, envs.g_lig, obs)
        
        # add noise to action
        key_dic1, key_dic2 = {}, {}
        for pdb in actions:
            key, nkey1, nkey2 = jax.random.split(key, 3)
            key_dic1[pdb], key_dic2[pdb] = nkey1, nkey2

        actions = jax.tree_util.tree_map(
                lambda x, y, z: jnp.concatenate(
                    (x[:4] + jnp.random.normal(y, shape=(3,))*rot_sd,
                     x[4:] + jnp.random.normal(z, shape=(3,))*tr_sd),
                    axis=-1),
                actions, key_dic1, key_dic2)

        # execute action and get reward
        rewards = envs.step(actions)
        
        # observe next status
        next_obs = envs.get_state()

        # store experiences in replay buffer
        experiences = jax.tree_util.tree_map(
                lambda a, b, c, d, e: Experience(a, b, c, d, e),
                obs, next_obs, actions, rewards, confidence)
        r_buffer.add(experiences)

        # proceed to next step
        obs = next_obs

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
        if args.steps % args.episode: envs.reset(dataset)
