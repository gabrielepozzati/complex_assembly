# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_action_jaxpy
import argparse
import os
import random
import time
from distutils.util import strtobool
from typing import Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pybullet_envs  # noqa
from flax.training.train_state import TrainState
from stable_baselines3.common.buffers import ReplayBuffer


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
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
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=25e3,
        help="timestep to start learning")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    args = parser.parse_args()
    # fmt: on
    return args


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, actor_key, qf1_key = jax.random.split(key, 3)

    # env setup
    envs = DockingEnv(dataset, key)

    r_buffer = ReplayBuffer(
        args.buffer_size,
        device="cpu")

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    actor = Actor()
    crit = QNetwork()
    
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(),
        target_params=actor.init(),
        tx=optax.adam(learning_rate=args.learning_rate))

    crit_state = TrainState.create(
        apply_fn=crit.apply,
        params=crit.init(),
        target_params=crit.init(),
        tx=optax.adam(learning_rate=args.learning_rate))

    actor.apply = jax.jit(actor.apply)
    crit.apply = jax.jit(crit.apply)

    @jax.jit
    def update_critic(
        actor_state: TrainState,
        crit_state: TrainState,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray):

        next_action = actor.apply(actor_state.target_params, next_state)
        next_q = crit.apply(crit_state.target_params, next_state, next_action)
        y = reward + args.gamma * (next_q)

        def mse_loss(params):
            q = crit.apply(params, state, action)
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
            return -crit.apply(crit_state.params, state, actor.apply(params, state)).mean()

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

    start_time = time.time()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions = actor.apply(actor_state.params, obs)
            actions = np.array(
                [
                    (
                        jax.device_get(actions)[0] + np.random.normal(action_bias, action_scale * args.exploration_noise)[0]
                    ).clip(envs.single_action_space.low, envs.single_action_space.high)
                ]
            )

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            qf1_state, qf1_loss_value, qf1_a_values = update_critic(
                actor_state,
                qf1_state,
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

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss_value.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss_value.item(), global_step)
                writer.add_scalar("losses/qf1_values", qf1_a_values.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
