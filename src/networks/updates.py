import jax
import json
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

path = '/'.join(__file__.split('/')[:-3])
with open(path+'/src/config.json') as j: config = json.load(j)

def actor_fw(mask, nodes, edges, i, j):
    actor = Actor(config)
    return actor(mask, nodes, edges, i, j)

def critic_fw(mask, nodes, edges, i, j, action):
    critic = Critic(config)
    return critic(mask, nodes, edges, i, j, action)

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

