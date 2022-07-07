import os
import sys
import copy
import time
import pickle
import numbers
import logging
import optax

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

from model import *
from haiku import avg_pool
from coordinates import *

class GradientUpdater:
    """A stateless abstraction around an init_fn/update_fn pair.
    This extracts some common boilerplate from the training loop.
    """
    def __init__(self, net_init, loss_fn,
                 optimizer: optax.GradientTransformation):
        self._net_init = net_init
        self._loss_fn = loss_fn
        self._opt = optimizer
        
    @functools.partial(jax.jit, static_argnums=0)
    def init(self, master_rng, data):
        """Initializes state of the updater."""
        out_rng, init_rng = jax.random.split(master_rng)
        params = self._net_init(init_rng, data)
        opt_state = self._opt.init(params)
        out = dict(
            step=np.array(0),
            rng=out_rng,
            opt_state=opt_state,
            params=params,
        )
        return out

    @functools.partial(jax.jit, static_argnums=0)
    def update(self, state: Mapping[str, Any], data: Mapping[str, jnp.ndarray]):
        """Updates the state using some data and returns metrics."""
        rng, new_rng = jax.random.split(state['rng'])
        params = state['params']
        loss, g = jax.value_and_grad(self._loss_fn)(params, rng, data)
        updates, opt_state = self._opt.update(g, state['opt_state'])
        params = optax.apply_updates(params, updates)

        new_state = {
            'step': state['step'] + 1,
            'rng': new_rng,
            'opt_state': opt_state,
            'params': params,
        }

        metrics = {
            'step': state['step'],
            'loss': loss,
        }
        return new_state, metrics

def loss_fn(forward_fn,
            params,
            rng,
            data: Mapping[str, jnp.ndarray],
            is_training: bool = True) -> jnp.ndarray:
    """Compute the loss on data wrt params."""

    pooled_matches, pooled_indexes = forward_fn(params, rng, data)
    #id_idx, rt_idx = pooled_indexes[jnp.argmax(pooled_matches)]
    loss = 1 - (jnp.max(pooled_matches))
    return loss

def build_forward_fn(num_heads: int):
    """Create the model's forward pass."""

    def forward_fn(data: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
        """Forward pass."""
        
        progress_track = [0 for chain in data]

        encoder = MultiHeadAttention(num_heads=num_heads, key_size=8)
        matcher = MultiHeadAttention(num_heads=num_heads, key_size=8)
        
        # define clouds
        clouds = []
        for chain in data['chains']:
            idxs = jnp.argwhere(data['surface_mask'][chain]==1 or \
                                data['ca_mask'][chain]==1)

            clouds.append(data['chains'][chain][idxs])

        # embed surface clouds
        embeddings = [encoder(chain, chain, chain) for chain in clouds]
        updates = copy.deepcopy(embeddings)

        # compare embeddings all-vs-all
        matches = []
        for emb1 in updates:
            match_line = [matcher(emb1, emb2, emb2) for emb2 in embeddings]
            matches.append(match_line)
        
        while jnp.any(progress_track == 0):
            # select cloud pair according to embeddings
            pooled_indexes = []
            pooled_matches = []
            for idx1 in range(len(embeddings)):
                for idx2 in range(len(embeddings)):
                    if idx1 > idx2: continue
                    if idx1 != idx2:
                        tensors = [matches[idx1][idx2], matches[idx2][idx1]]
                        tensor = jnp.concatenate(tensors, axis=0)
                    else: 
                        tensor = matches[idx1][idx2]

                    pooled_matches.append(avg_pool(tensor, tensor.shape, strides=1, padding='VALID'))
                    pooled_indexes.append([idx1, idx2])
        
            pooled_matches = jax.nn.softmax(jnp.concatenate(pooled_matches))
            id_idx, tr_idx = pooled_indexes[jnp.argmax(pooled_matches)]

            # max contacts and min clashes between sampled structures
            

            #cmap = jnp.sqrt(jnp.sum((clouds[id_idx]**2, jnp.transpose(clouds[tr_idx]**2)), axis=-1))
            #interface = jnp.argwhere(cmap<8)
            #surface = jnp.argwhere(cmap>8)



        return pooled_matches, pooled_indexes
    return forward_fn

def main():
    num_heads = 8
    MAX_STEPS = 1000
    grad_clip_value = 0.5
    learning_rate = 0.05

    # Create the dataset.
    path = '/proj/berzelius-2021-29/users/x_gabpo/complex_assembly/data/train_features.pkl'
    with open(path, 'rb') as data: dataset = pickle.load(data)
    pdb = list(dataset.keys())[0]

    # Set up the model, loss, and updater.
    forward_fn = build_forward_fn(num_heads)
    forward_fn = hk.transform(forward_fn)
    
    loss = functools.partial(loss_fn, forward_fn.apply)
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_value),
        optax.adam(learning_rate, b1=0.9, b2=0.99))
    
    updater = GradientUpdater(forward_fn.init, loss, optimizer)
    
    # Initialize parameters.
    logging.info('Initializing parameters...')
    rng = jax.random.PRNGKey(42)
    state = updater.init(rng, dataset[pdb])
    logging.info('Starting train loop...')
    prev_time = time.time()
    for step in range(MAX_STEPS):
        print (f'Epoch {step} ...')
        loss_acc = 0
        for pdb in dataset:
            if len(dataset[pdb]['chains'].keys()) == 1: continue
            state, metrics = updater.update(state, dataset[pdb])
            if step == 0: print (pdb)
            loss_acc += metrics['loss']
        print ('Loss:', loss_acc/len(dataset.keys()))

if __name__ == '__main__':
    main() 
