import os
import sys
import pickle

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

from model import *
from haiku import GRU, avg_pool
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

    pooled_matches, pooled_indexes = forward_fn(params, rng, data, is_training)
    #id_idx, rt_idx = pooled_indexes[jnp.argmax(pooled_matches)]
    pooled_matches = pooled_matches + jnp.min(pooled_matches)
    pooled_matches = pooled_matches / jnp.sum(pooled_matches)
    loss = 1 - (jnp.max(pooled_matches)
    return loss

def build_forward_fn(num_heads: int):
    """Create the model's forward pass."""

    def forward_fn(data: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
        """Forward pass."""
        
        data = initialize_clouds(data, 42)        
        chain_seeds = copy.deepcopy(data)
        chain_number = len(data.keys())

        encoder = MultiHeadAttention(num_heads=num_heads, key_size=8)
        matcher = MultiHeadAttention(num_heads=num_heads, key_size=8)

        # embed surface clouds
        embeddings = [encoder(data[chain], data[chain], data[chain]) for chain in data]
        updates = copy.deepcopy(embeddings)

        # compare embeddings all-vs-all
        matches = []
        for emb1 in updates:
            match_line = [matcher(emb1, emb2, emb2) for emb2 in embeddings]
            matches.append(match_line)
        
        pooled_indexes = []
        pooled_matches = []
        for idx1 in range(len(embeddings)):
            for idx2 in range(len(embeddings)):
                if idx1 > idx2: continue
                if idx1 != idx2:
                    tensors = [matches[idx1][idx2], jnp.transpose(matches[idx2][idx1])]
                    tensor = jnp.concatenate(tensors, axis=1)
                else: 
                    tensor = matches[idx1][idx2]
                pooled_matches.append(avg_pool(tensor, tensor.shape))
                pooled_indexes.append(idx1, idx2)
        pooled_matches = jnp.concatenate(pooled_matches)

        return pooled_matches, pooled_indexes
    return forward_fn

def main():
    # Create the dataset.
    with open('', 'rb') as data: dataset = pkl.load(data)

    # Set up the model, loss, and updater.
    forward_fn = build_forward_fn(num_heads, num_layers, dropout_rate)
    forward_fn = hk.transform(forward_fn)
    
    loss_fn = functools.partial(lm_loss_fn, forward_fn.apply)
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_value),
        optax.adam(learning_rate, b1=0.9, b2=0.99))
    
    updater = GradientUpdater(forward_fn.init, loss_fn, optimizer)
    
    # Initialize parameters.
    logging.info('Initializing parameters...')
    rng = jax.random.PRNGKey(42)
    data = next(train_dataset)
    state = updater.init(rng, data)
    logging.info('Starting train loop...')
    prev_time = time.time()
    for step in range(MAX_STEPS):
        for pdb in dataset:
            state, metrics = updater.update(state, dataset[pdb]['chain'])

if __name__ == '__main__':
    main() 
