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

from modules import *
from haiku import avg_pool
from coordinates import *
from graph import *
from jraph._src.models import *
from ops import *

class GradientUpdater:
    """A stateless abstraction around an init_fn/update_fn pair.
    This extracts some common boilerplate from the training loop.
    """
    def __init__(self, net_init, loss_fn,
                 optimizer: optax.GradientTransformation):
        self._net_init = net_init
        self._loss_fn = loss_fn
        self._opt = optimizer
        
    def init(self, master_rng, data):
        """Initializes state of the updater."""
        masks = data['masks']
        nodes = data['nodes']
        clouds = data['clouds']
        out_rng, init_rng = jax.random.split(master_rng)
        params = self._net_init(init_rng, clouds, nodes, masks)
        opt_state = self._opt.init(params)
        out = dict(
            step=np.array(0),
            rng=out_rng,
            opt_state=opt_state,
            params=params,
        )
        return out

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

    masks = data['masks']
    nodes = data['nodes']
    clouds = data['clouds']

    clouds_p, tracks = forward_fn(params, rng, clouds, nodes, masks)
    
    sen = data['igraph'].senders
    rec = data['igraph'].receivers
    edges = [[i, j] for i, j in zip(sen, rec)]

    all_clashes = 0
    bad_contacts = 0
    good_contacts = 0
    for cloud, track in zip(clouds, tracks):
        cmap = cmap_from_2D(cloud, cloud)
        cmap_breaks = [len(data['clouds'][idx]) for idx in track][:-1]

        e1, e2 = 0, 0
        for idx1, c_idx1 in enumerate(track):
            e1 += cmap_breaks[idx1]
            for idx2, c_idx2 in enumerate(track):
                e2 += cmap_breaks[idx2]
                if idx1 >= idx2: continue
                s1 = e1 - cmap_breaks[idx1]
                s2 = e2 - cmap_breaks[idx2]
                if e1 < len(cmap) and e2 < len(cmap):
                    sub_cmap = cmap[s1:e1,s2:e2]
                elif e1 == len(cmap):
                    sub_cmap = cmap[s1:,s2:e2]
                elif e2 == len(cmap):
                    sub_cmap = cmap[s1:e1,s2:]                

                contacts = jnp.where(sub_cmap<8, 1, 0)
                clashes = jnp.where(sub_cmap<3, 0, 1)
                all_clashes += jnp.sum(1-clashes)
                contacts *= clashes

                if [c_idx1, c_idx2] in edges:
                    good_contacts += jnp.sum(contacts)
                else: 
                    bad_contacts += jnp.sum(contacts)
            e2 = 0

    all_clashes = jnp.log10(all_clashes)
    bad_contacts = jnp.log10(bad_contacts)
    good_contacts = 1/jnp.log10(good_contacts)
    return all_clashes+bad_contacts+good_contacts

def build_forward_fn(num_heads: int, num_channels: int):
    """Create the model's forward pass."""

    def forward_fn(clouds, nodes, edges, masks) -> jnp.ndarray:
        
        """Forward pass."""
        ################################
        ##### layers/modules definitions
        #base node/edge encoding
        n_enc1 = Linear(num_channels**2, num_input_dims=2)
        n_enc2 = MultiLayerPerceptron(num_channels, num_channels, 2)
        e_enc = MultiLayerPerceptron(num_channels, num_channels, 3)

        # surface encoding 
        single_enc1 = GraphUpdates(num_channels)
        single_enc2 = GraphUpdates(num_channels*2)
        rec_summary = Linear(1)

        # comparison
        pairs_enc1 = RelativeSurfaceEncoding(num_channels, num_heads)
        pairs_enc2 = GraphUpdates(num_channels)
        pairs_enc3 = RelativeSurfaceEncoding(num_channels, num_heads)
        pairs_enc4 = GraphUpdates(num_channels)
        lig_summary = Linear(1)

        # node selection
        nodes_selection = GraphUpdates(num_channels)
        node_summary = Linear(1)

        # IPA module to include euclidean info into graph
        IPA = InvariantPointAttention(num_heads, num_channels, num_channels, 
                                      num_channels, num_channels, num_channels)

        # transformation graph updates
        docker = GraphUpdates(num_channels)
        rotator = Linear(3)
        traslator = Linear(3)

        #######################
        ##### main network body

        # initialize tensors
        nodes = jax.nn.relu(n_enc1(nodes))
        nodes = jax.nn.relu(n_enc2(nodes))
        
        node_updates, summary = [], []
        edges, senders, receivers = [], [], []
        for idx in range(masks.shape[0]):
            edges_idx, senders_idx, receivers_idx = \
                get_closest_edges(clouds[idx], masks[idx], e_enc)
            
            g = jraph.GraphsTuple(
                nodes=nodes[idx], edges=edges_idx,
                senders=senders_idx, receivers=receivers_idx,
                n_node=jnp.array([len(nodes[idx])]), 
                n_edge=jnp.array([len(edges)]), 
                globals=jnp.zeros(num_channels))
            g = single_enc1(g)
            g = single_enc2(g)

            edges.append(edges_idx)
            senders.append(senders_idx)
            receivers.append(receivers_idx)
            summary.append(rec_summary(g.globals))
            node_updates.append(g.nodes)

        edges = jnp.concatenate(edges)
        senders = jnp.concatenate(senders)
        receivers = jnp.concatenate(receivers)
        nodes = jnp.concatenate(node_updates)
        summary = jnp.concatenate(summary)

        # pick best receptor
        choice_rec = jax.nn.softmax(summary)
        idx_rec = jnp.argmax(choice_rec)
        c_rec = clouds[idx_rec]
        g_rec = jraph.GraphsTuple(
            nodes=nodes[idx_rec], edges=edges[idx_rec],
            senders=senders[idx_rec], receivers=receivers[idx_rec],
            n_node=jnp.array([len(nodes[idx_rec])]), 
            n_edge=jnp.array([len(edges[idx_rec])]),
            globals=jnp.zeros(num_channels))

        # pick best ligand
        choice_lig = []
        nodes_rec, nodes_lig = [], []
        edges_rec, edges_lig = [], []
        for idx in range(masks.shape[0]):
            # compare ligand nodes with receptor nodes
            g_lig = jraph.GraphsTuple(
              nodes=nodes[idx], edges=edges[idx],
              senders=senders[idx], receivers=receivers[idx],
              n_node=jnp.array([len(nodes[idx])]),
              n_edge=jnp.array([len(edges[idx])]),
              globals=jnp.zeros(num_channels))

            up_n_rec = pairs_enc1(g_rec.nodes, g_lig.nodes)
            up_g_rec = g_rec._replace(nodes=up_n_rec)
            up_g_rec = pairs_enc2(up_g_rec)

            up_n_lig = pairs_enc3(g_lig.nodes, up_n_rec)
            up_g_lig = g_lig._replace(nodes=up_n_lig)
            up_g_lig = pairs_enc4(up_g_lig)

            nodes_rec.append(up_g_rec.nodes)
            edges_rec.append(up_g_rec.edges)
            nodes_lig.append(up_g_lig.nodes)
            edges_lig.append(up_g_lig.edges)
            choice_lig.append(jax.nn.relu(lig_summary(g_lig.globals)))

        nodes_rec = jnp.concatenate(nodes_rec)
        edges_rec = jnp.concatenate(edges_rec)
        nodes_lig = jnp.concatenate(nodes_lig)
        edges_lig = jnp.concatenate(edges_lig)
        choice_lig = jnp.concatenate(choice_lig)
        choice_lig = jax.nn.softmax(choice_lig)
        idx_lig = jnp.argmax(lig_choice)
 
        c_rec = clouds[idx_rec]
        c_lig = clouds[idx_lig]

        # subsample nodes
        g_rec = jraph.GraphsTuple(
            nodes=nodes_rec[idx_rec], edges=edges_rec[idx_rec],
            senders=senders[idx_rec], receivers=receivers[idx_rec],
            n_node=jnp.array([len(nodes_rec[idx_rec])]),
            n_edge=jnp.array([len(edges_rec[idx_rec])]),
            globals=jnp.zeros(num_channels))
        g_rec = nodes_selection(g_rec)
        n_rec = node_summary(g_rec.nodes)
        
        g_lig = jraph.GraphsTuple(
            nodes=nodes_lig[idx_lig], edges=edges_lig[idx_lig],
            senders=senders[idx_lig], receivers=receivers[idx_lig],
            n_node=jnp.array([len(nodes_lig[idx_lig])]),
            n_edge=jnp.array([len(edges_lig[idx_lig])]),
            globals=jnp.zeros(num_channels))
        g_lig = nodes_selection(g_lig)
        n_lig = node_summary(g_lig.nodes)


        print ('Init docking')
        # initialize nodes and clouds 1D sequences of length rec_CA+lig_CA
        all_nodes = jnp.concatenate([n_rec, n_lig], axis=0)
        all_points = jnp.concatenate([c_rec, c_lig], axis=0)
        print (f'Nodes: {n_rec.shape} {n_lig.shape}')
        print (f'Clouds: {c_rec.shape} {c_lig.shape}')

        # initialize 2D features with shape = (rec_CA+lig_CA, rec_CA+lig_CA)
        e_rec = jnp.reshape(e_rec, [n_rec.shape[0], n_rec.shape[0], num_channels])
        e_lig = jnp.reshape(e_lig, [n_lig.shape[0], n_lig.shape[0], num_channels])
            
        filler = jnp.zeros([n_rec.shape[0], n_lig.shape[0], 16])
        all_r0 = jnp.concatenate([e_rec, filler], axis=1)
        filler = jnp.swapaxes(filler, -2, -3)
        all_l0 = jnp.concatenate([filler, e_lig], axis=1)
        all_edges = jnp.concatenate([all_r0, all_l0], axis=0)

        # docking step
        for _ in range(10):
            print (f'Docking iteration {_}')
            # invariant point attention
            all_nodes = IPA(all_nodes, all_edges, all_points)
            n_rec, n_lig = jnp.split(all_nodes, [n_rec.shape[0]], axis=0)

            # setting up graph with new features
            e_lig = jnp.reshape(e_lig, [n_lig.shape[0]**2, num_channels])

            g_out = jraph.GraphsTuple(
                nodes=n_lig, edges=e_lig,
                senders=g_lig.senders, receivers=g_lig.receivers,
                n_node=g_lig.n_node, n_edge=g_lig.n_edge,
                globals=g_lig.globals)

            # aggregation
            g_out = aggregation3(g_out)

            # computing T
            rot = jax.nn.softmax(rotator(g_out.globals))
            rot = rot_from_pred(jnp.ravel(rot))
            trasl = traslator(g_out.globals)
                
            # updated cloud
            c_lig = jnp.matmul(rot, jnp.transpose(c_lig))
            c_lig = jnp.transpose(c_lig) + trasl
            all_points = jnp.concatenate([c_rec, c_lig], axis=0)

            # updated 2D features
            cmap = jnp.sqrt(jnp.sum(
                (all_points[:, None, :]-all_points[None, :, :])**2, axis=-1))
            cmap = jnp.expand_dims(cmap, axis=-1)
            all_edges = one_hot_cmap(cmap)
                
            # embed 2D features
            all_edges = e_enc(all_edges)

        return
    return forward_fn

def main():
    # Run parameters
    num_heads = 8
    num_channels = 16
    MAX_STEPS = 1000
    grad_clip_value = 0.5
    learning_rate = 0.1

    # Load the dataset.
    path = '/home/pozzati/complex_assembly/data/train_features.pkl'
    with open(path, 'rb') as data: dataset = pickle.load(data)
    pdb = list(dataset.keys())[0]

    # Set up the model
    forward_fn = build_forward_fn(num_heads, num_channels)
    fw_init = jax.jit(hk.transform(forward_fn).init)
    fw_apply = jax.jit(hk.transform(forward_fn).apply)

    # Set up the loss
    loss = functools.partial(loss_fn, fw_apply)
    
    # Set up the updater
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_value),
        optax.adam(learning_rate, b1=0.9, b2=0.99)) 
    updater = GradientUpdater(fw_init, loss, optimizer)
    
    # Initialize parameters.
    for pdb in dataset: del dataset[pdb]['mapping']

    rng = jax.random.PRNGKey(42)
    pdb = list(dataset.keys())[0]
    state = updater.init(rng, dataset[pdb])

    for step in range(MAX_STEPS):
        print (f'Epoch {step} ...')
        count = 0
        loss_acc = 0
        for pdb in dataset:
            state, metrics = updater.update(state, dataset[pdb])
            if step == 0: print (pdb)
            loss_acc += metrics['loss']
            count += 1
            if count == 10: break
        print ('Loss:', loss_acc/count)

if __name__ == '__main__':
    main() 
