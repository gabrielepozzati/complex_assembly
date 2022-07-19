import os
import sys
import copy
import time
import jraph
import pickle
import numbers
import logging
import optax

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

    out_list = forward_fn(params, rng, data)
    
    avg = 0
    d_avg = 0
    c_avg = 0 
    for id_idx, tr_idx, q, tr in out_list:

        b, c, d = q / jnp.sqrt(jnp.sum(q**2)+1)
        a = 1 / jnp.sqrt(jnp.sum(q**2)+1)
        aa, bb, cc, dd = a**2, b**2, c**2, d**2

        rot = jnp.array(
            [[aa+bb-cc-dd, 
              2*b*c-2*a*d,
              2*b*d+2*a*c],
         
             [2*b*c+2*a*d,
              aa-bb+cc-dd,
              2*c*d-2*a*b],

             [2*b*d-2*a*c,
              2*c*d+2*a*b,
              aa-bb-cc+dd]
            ])

        id_cloud = data[id_idx]
        tr_cloud = data[tr_idx]
        tr_cloud = jnp.matmul(rot, jnp.transpose(tr_cloud))
        tr_cloud = jnp.transpose(tr_cloud) + tr
        
        id_len = id_cloud.shape[0]
        tr_len = tr_cloud.shape[0]
        id_cloud = jnp.expand_dims(id_cloud, axis=1)
        tr_cloud = jnp.expand_dims(tr_cloud, axis=0)
        id_cloud = jnp.broadcast_to(id_cloud, (id_len, tr_len, 3))
        tr_cloud = jnp.broadcast_to(tr_cloud, (id_len, tr_len, 3))
        cmap = jnp.sqrt(jnp.sum((id_cloud-tr_cloud)**2, axis=-1))
        clash_loss = jnp.sum(jnp.where(cmap<3, 3-cmap, 0))#/max(id_cloud.shape[0], tr_cloud.shape[1])
        distance_loss = jnp.maximum(jnp.min(cmap)-4, 0)
        avg += (distance_loss + clash_loss)
        d_avg += distance_loss
        c_avg += clash_loss

    return avg/len(out_list)

def build_forward_fn(num_heads: int):
    """Create the model's forward pass."""

    def forward_fn(data) -> jnp.ndarray:
        """Forward pass."""
        # set network params
        ks = 8
        contact_thr = 8
        chain_num = len(clouds)
        progress_track = [0 for chain in clouds]
        
        

        encoder = MultiHeadAttention(num_heads=num_heads, key_size=ks)
        matcher = MultiHeadAttention(num_heads=num_heads, key_size=ks)
        docker1 = MultiHeadAttention(num_heads=num_heads, key_size=ks*ks)
        docker2 = Linear(ks*ks)
        rotator = Linear(3)
        traslator = Linear(3)
        norm1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        norm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

        # define clouds
#        cmaps = []
#        clouds = []
#        graphs = []
#        for chain in data['chains'].keys():
#            # filter all surface atoms and non-surface CA
#            idxs = jnp.argwhere(data['surface_mask'][chain]==1 or \
#                                data['ca_mask'][chain]==1)
#            cloud = data['chains'][chain][idxs]
#            clouds.append(cloud)

            # compute cmaps
#            cmap = jnp.sqrt(jnp.sum((cloud**2, jnp.transpose(cloud**2)), axis=-1))
#            cmaps.append(cmap)

            # compute graphs
#            n_node = jnp.array(len(cloud))
#            
#            senders = jnp.array([idx1 for idx1 in range(chain_num) \
#                for idx2 in jnp.argwhere(cmap[idx1]<contact_thr))])
#            
#            receivers = jnp.array([idx2 for idx1 in range(chain_num) \
#                for idx2 in jnp.argwhere(cmap[idx1]<contact_thr))])
#
#            edges = jnp.array([cmap[idx1][idx2] for idx1 in range(chain_num) \
#                for idx2 in jnp.argwhere(cmap[idx1]<contact_thr)])
#
#            n_edge = jnp.array(len(senders))
#
#            context = 
        # embed surface clouds
        embeddings = [jax.nn.softmax(encoder(chain, chain, chain)) for chain in clouds]
        updates = copy.deepcopy(embeddings)


        # compare embeddings all-vs-all
        matches = []
        for emb1 in updates:
            match_line = [jax.nn.softmax(matcher(emb1, emb2, emb2)) for emb2 in embeddings]
            matches.append(match_line)
#        while jnp.any(progress_track == 0):
#        # select cloud pair according to embeddings
#            pooled_indexes = []
#            pooled_matches = []
#            for idx1 in range(len(embeddings)):
#                for idx2 in range(len(embeddings)):
#                    if idx1 > idx2: continue
#                    if idx1 != idx2:
#                        tensors = [matches[idx1][idx2], matches[idx2][idx1]]
#                        tensor = jnp.concatenate(tensors, axis=0)
#                    else: 
#                        tensor = matches[idx1][idx2]
#    
#                    pooled_matches.append(avg_pool(tensor, tensor.shape, strides=1, padding='VALID'))
#                    pooled_indexes.append([idx1, idx2])
#            
#            pooled_matches = jax.nn.softmax(jnp.concatenate(pooled_matches))
#            id_idx, tr_idx = pooled_indexes[jnp.argmax(pooled_matches)]
#            id_emb, tr_emb = embeddings[id_idx], embeddings[tr_idx]
        
        out_list = []
        for id_idx, id_emb in enumerate(embeddings):
            for tr_idx, tr_emb in enumerate(embeddings):

                # max contacts and min clashes between sampled structures
                tr_emb = jax.nn.softmax(docker1(tr_emb, id_emb, id_emb)) 
                tr_emb = norm1(tr_emb)
                tr_emb = jnp.einsum('ab, ab -> b', tr_emb, tr_emb)

                tr_emb = jax.nn.softmax(docker2(tr_emb))
                tr_emb = norm2(tr_emb)
                
                rot = jax.nn.softmax(rotator(tr_emb))
                tr = traslator(tr_emb)
                out_list.append([id_idx, tr_idx, rot, tr])

        return out_list
    return forward_fn

def example_generator(dataset):
    graphs = ()
    id_transform = [1,0,0,0,0,0,0]
    for pdb in dataset:
        cloud = dataset[pdb]['cloud']
        total_len = cloud.shape[0]

        # unpack complex-wise data
        node_feats = dataset[pdb]['nodes']
        edge_feats = dataset[pdb]['edges']
        out_edges = dataset[pdb]['sender']
        in_edges = dataset[pdb]['receiver']
        
        edge_labels = dataset[pdb]['labels']
        surface_mask = dataset[pdb]['smask']
        id_mask = dataset[pdb]['imask']

        # define chain boundaries
        split_idx = []
        prev_chain = ''
        for idx, (sid, mid, cid, rid) in enumerate(id_mask):
            if prev_chain == '': prev_chain = cid
            if cid != prev_chain: split_idx.append(idx)
            prev_chain = cid

        # split point clouds by chain
        clouds = jnp.split(cloud, split_idx)

        # split graph-features by chain
        nodes = jnp.split(nodes, split_idx)
        
        first_idx = 0
        in_edges_single = []
        out_edges_single = []
        edge_feats_single = []
        for last_idx in enumerate(split_idx+[total_len]):
            for idx in enumerate(out_edges): 
                if out_edges[idx] in range(first_idx, last_idx) \
                and in_edges[idx] in range(first_idx, last_idx):
                    in_edges_single.append(in_edges[idx])
                    out_edges_single.append(out_edges[idx])
                    edge_feats_single.append(edge_feats[idx])
            first_idx = last_idx
            graphs += (
                jraph.GraphsTuple(
                    nodes=nodes_single, senders=out_edges_single,
                    receivers=in_edges_single, edges=edge_feats_single,
                    n_node=len(node_single), n_edge=len(edge_feats_single), 
                    globals=id_transform))

        yield (clouds, graphs)



def main():
    # Run parameters
    num_heads = 8
    MAX_STEPS = 1000
    grad_clip_value = 1
    learning_rate = 0.1

    # Load the dataset.
    path = '/proj/berzelius-2021-29/users/x_gabpo/complex_assembly/data/train_features.pkl'
    with open(path, 'rb') as data: dataset = pickle.load(data)
    pdb = list(dataset.keys())[0]

    # Set up the model
    forward_fn = build_forward_fn(num_heads)
    forward_fn = hk.transform(forward_fn)
    
    # Set up the loss
    loss = functools.partial(loss_fn, forward_fn.apply)
    
    # Set up the updater
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_value),
        optax.adam(learning_rate, b1=0.9, b2=0.99)) 
    updater = GradientUpdater(forward_fn.init, loss, optimizer)
    
    # Initialize parameters.
    rng = jax.random.PRNGKey(42)
    state = updater.init(rng, example)

    
    for step in range(MAX_STEPS):
        print (f'Epoch {step} ...')
        count = 0
        loss_acc = 0
        for pdb in dataset:
            if len(dataset[pdb]['chains'].keys()) == 1: continue
            example = example_packer(dataset[pdb])
            state, metrics = updater.update(state, example)
            if step == 0: print (pdb)
            loss_acc += metrics['loss']
            count += 1
            if count == 10: break
        print ('Loss:', loss_acc/count)

if __name__ == '__main__':
    main() 
