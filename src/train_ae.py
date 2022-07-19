import os
import sys
import copy
import time
import jraph
import pickle
import numbers
import logging
import optax
import pdb as debugger
import pickle as pkl

import numpy as np
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
            data,
            is_training: bool = True) -> jnp.ndarray:
    """Compute the loss on data wrt params."""

    types, asa = jnp.split(data, [11], axis=-1)
    out_types, out_asa = forward_fn(params, rng, data)
    out_types += 0.1**20

    atm_crossent = -(jnp.sum(types*jnp.log(out_types), axis=-1))
    res_crossent = jnp.sum(atm_crossent, axis=-1)/16
    seq_crossent = jnp.sum(res_crossent)/res_crossent.shape[0]
    
    asa = jnp.reshape(asa, asa.shape[:2])    
    res_rmse = jnp.sqrt(jnp.sum((asa-out_asa)**2, axis=-1)/16)
    seq_rmse = jnp.sum(res_rmse)/res_rmse.shape[0]
    
    return seq_crossent+seq_rmse

def build_forward_fn():
    """Create the model's forward pass."""

    def forward_fn(nodes) -> jnp.ndarray:
        """Forward pass."""
        
        enc_space = 16
        multipliers = 3
        layer_stack = [Linear(enc_space)]
        for n in range(1, multipliers+1):
            if n == multipliers: dims = 2
            else: dims = 1
            layer_stack = [Linear(enc_space*2*multipliers, num_input_dims=dims)] + layer_stack
            layer_stack = layer_stack + [Linear(enc_space*2*multipliers)]
        restore1 = Linear((16,11))
        restore2 = Linear(16)
        #norm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
       
        for layer in layer_stack: nodes = jax.nn.relu(layer(nodes))
        types = jax.nn.softmax(restore1(nodes))
        asa = restore2(nodes)

        return [types, asa]
    return forward_fn

def example_packer(data):
    clouds = []
    for chain in data['chains'].keys():
        # filter all surface atoms and non-surface CA
        idxs = jnp.argwhere(data['surface_mask'][chain]==1)

        idxs = jnp.concatenate((idxs, jnp.argwhere(data['ca_mask'][chain]==1)))
        cloud = data['chains'][chain][idxs]
        clouds.append(jnp.reshape(cloud, (-1,3)))
    return clouds

def main():
    MAX_STEPS = 2000
    grad_clip_value = 0.5
    learning_rate = 0.001
    val_split = 20

    # Create the dataset.
    path = '/proj/berzelius-2021-29/users/x_gabpo/complex_assembly/data/train_features.pkl'
    with open(path, 'rb') as data: dataset = pickle.load(data)
    pdb = list(dataset.keys())[0]

    # Set up the model, loss, and updater.
    forward_fn = build_forward_fn()
    forward_fn = hk.transform(forward_fn)
    
    loss = functools.partial(loss_fn, forward_fn.apply)
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_value),
        optax.adam(learning_rate, b1=0.9, b2=0.99))
    
    updater = GradientUpdater(forward_fn.init, loss, optimizer)
    
    # Initialize parameters.
    logging.info('Initializing parameters...')
    rng = jax.random.PRNGKey(42)
    example = dataset[pdb]['nodes']
    state = updater.init(rng, example)

    logging.info('Starting train loop...')
    prev_time = time.time()
    for step in range(MAX_STEPS):
        print (f'Epoch {step} ...')
        if step % 500 == 0: learning_rate *= 0.5
        # train epoch
        loss_acc = 0
        for idx, pdb in enumerate(dataset):
            if idx < val_split: continue
            example = dataset[pdb]['nodes']
            state, metrics = updater.update(state, example)
            if step == 0: print (pdb)
            loss_acc += metrics['loss']

        # validation epoch
        ce_acc, rmse_acc = 0, 0
        for idx, pdb in enumerate(dataset):
            example = dataset[pdb]['nodes']
            types, asa = jnp.split(example, [11], axis=-1)
            out_types, out_asa = forward_fn.apply(state['params'], rng, example)
            out_types += 0.1**20

            atm_crossent = -(jnp.sum(types*jnp.log(out_types), axis=-1))
            res_crossent = jnp.sum(atm_crossent, axis=-1)/16
            seq_crossent = jnp.sum(res_crossent)/res_crossent.shape[0]

            asa = jnp.reshape(asa, asa.shape[:2])
            res_rmse = jnp.sqrt(jnp.sum((asa-out_asa)**2, axis=-1)/16)
            seq_rmse = jnp.sum(res_rmse)/res_rmse.shape[0]
            
            ce_acc += seq_crossent
            rmse_acc += seq_rmse

            if idx == val_split: break

        loss = round(loss_acc/(len(list(dataset.keys()))-val_split), 3)
        ce_loss = round(ce_acc/val_split, 3)
        rmse_loss = round(rmse_acc/val_split, 3)
        print ('TrainingLoss:', loss, '\n' 
               'Val.Loss:',ce_loss+rmse_loss, 'Val.CELoss:', ce_loss,'Val.RMSELoss:', rmse_loss)

    with open('ae.pkl', 'wb') as out:
        pkl.dump(state['params'], out)

if __name__ == '__main__':
    main() 
