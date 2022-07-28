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
from features import one_hot_cmap
from graph import *
from jraph._src.models import *


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

    clouds, tracks = forward_fn(params, rng, data)
    
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

    def forward_fn(data) -> jnp.ndarray:
        """Forward pass."""
        ################################
        ##### layers/modules definitions
        
        # node encoding
        n_enc1 = Linear(num_channels**2, num_input_dims=2)
        n_enc2 = MultiLayerPerceptron(num_channels, num_channels, 2)

        # edge encoding
        e_enc = MultiLayerPerceptron(num_channels, num_channels, 3)

        # node convolution encoding
        neigh_nodes_update = MultiLayerPerceptron(num_channels, num_channels, 2)

        # graph aggregation encoding
        aggregation1 = GraphNetwork(
            MultiLayerPerceptron(num_channels, num_channels, 1),
            MultiLayerPerceptron(num_channels, num_channels, 1),
            update_global_fn=MultiLayerPerceptron(num_channels, num_channels, 1))

        # single chain info summary
        graph_summary = Linear(1)

        # ligand receptor comparison
        chain_comp = MultiHeadAttention(num_heads, num_channels)

        # ligand info aggregation
        comp_nodes_update = MultiLayerPerceptron(num_channels, num_channels, 2)
        comp_edges_update = MultiLayerPerceptron(num_channels, num_channels, 2)
        comp_glob_update = MultiLayerPerceptron(num_channels, 1, 2)

        # triangle updates for chain pairs
        #inter_tri_att = TriangleModule(num_heads, num_channels)
        
        # IPA module to include euclidean info into graph
        IPA = InvariantPointAttention(num_heads, num_channels, num_channels, 
                                      num_channels, num_channels, num_channels)

        # adjust IPA nodes data
        #post_ipa_proc = MultiHeadAttention(num_heads, num_channels)

        # transformation graph updates
        tr_nodes_update = MultiLayerPerceptron(num_channels, num_channels, 2)
        tr_edges_update = MultiLayerPerceptron(num_channels, num_channels, 2)
        tr_glob_update = MultiLayerPerceptron(num_channels, num_channels, 2)
        rotator = Linear(3)
        traslator = Linear(3)

        # state evaluation graph updates
        all_nodes_update = MultiLayerPerceptron(num_channels, num_channels, 2)
        all_edges_update = MultiLayerPerceptron(num_channels, num_channels, 2)
        all_glob_update = MultiLayerPerceptron(num_channels, num_channels, 2)
        global_state = Linear(2)

        #######################
        ##### main network body

        ch_clouds = data['clouds']
        up_clouds = data['clouds']
        ch_lbl, up_lbl = data['clabel'], data['clabel']
        ch_graphs, up_graphs = [], []
        for idx, g in enumerate(data['graphs']):
            # graph embedding
            g_n, g_e = g.nodes, g.edges
            g_n = jax.nn.relu(n_enc1(g_n))
            g_n = jax.nn.relu(n_enc2(g_n))
            g_e = jax.nn.relu(e_enc(g_e))
            data['graphs'][idx] = g._replace(nodes=g_n, edges=g_e)
            
            # graph aggregation
            g = aggregation1(g)

            ch_graphs.append(g)
            up_graphs.append(g)
       
        ch_base = [g for g in data['graphs']]
        up_base = [g for g in data['graphs']]

        # summarize receptor information
        choice = [jax.nn.relu(graph_summary(g.globals)) for g in up_graphs]
        choice = jnp.concatenate(choice)

        #focus = jnp.array((1 for _ in range(len(ch_clouds))))
        #focus /= jnp.sum(focus)

        # stop signal
        step = 0
        sequence = []
        incomplete = True
        while incomplete or step < len(ch_graphs)**2:    
            # pick receptor
            choice = jax.nn.softmax(choice)
            idx_rec = jnp.argmax(choice)
            e_rec = up_base[idx_rec].edges
            c_rec = up_clouds[idx_rec]
            g_rec = up_graphs[idx_rec]
            n_rec = g_rec.nodes

            # screen for best ligand
            lig_choice = []
            for g_lig in ch_graphs+up_graphs:
                # compare ligand nodes with receptor nodes
                n_lig = g_lig.nodes
                n_lig = inter_chain_comp(n_lig, n_rec, n_rec)
                
                # create graph decoy with rec-lig nodes update
                g_lig = jraph.GraphsTuple(
                    nodes=n_lig, edges=g_lig.edges,
                    senders=g_lig.senders, receivers=g_lig.receivers,
                    n_node=g_lig.n_node, n_edge=g_lig.n_edge,
                    globals=g_lig.globals)

                # aggregate rec-lig information over graph
                g_lig = GraphNetwork(
                    update_node_fn=comp_nodes_update,
                    update_edge_fn=comp_edges_update,
                    update_global_fn=comp_glob_update)(g_lig)
                lig_choice.append(jax.nn.relu(g_lig.globals))

            # pick ligand
            lig_choice = jax.nn.softmax(jax.concatenate(lig_choice))
            idx_lig = jnp.argmax(lig_choice)
            e_lig = ch_base+up_base[idx_lig].edges
            c_lig = ch_clouds+up_clouds[idx_lig]
            g_lig = ch_graphs+up_graphs[idx_lig]
            n_lig = g_lig.nodes

            # initialize nodes and clouds 1D sequences of length rec_CA+lig_CA
            all_nodes = jnp.concatenate([n_rec, n_lig], axis=0)
            all_points = jnp.concatenate([c_rec, c_lig], axis=0)

            # initialize 2D features with shape = (rec_CA+lig_CA, rec_CA+lig_CA)
            e_rec = jnp.reshape(e_rec, [g_rec.n_node, g_rec.n_node, 34])
            e_lig = jnp.reshape(e_lig, [g_lig.n_node, g_lig.n_node, 34])
            
            filler0 = jnp.zeros([g_rec.n_node, g_lig.n_node, 33])
            filler1 = jnp.ones([g_rec.n_node, g_lig.n_node, 1])
            filler = jnp.concatenate([filler0, filler1], axis=-1)
            
            all_r0 = jnp.concatenate([e_rec, filler], axis=0)
            filler = jnp.swapaxes(filler, -2, -3)
            all_l0 = jnp.concatenate([filler, e_lig], axis=0)
            all_edges = jnp.concatenate([all_r0, all_l0], axis=1)

            # docking steps
            for _ in range(10):
                # invariant point attention
                all_nodes = IPA(all_nodes, all_edges, all_points)
                n_rec, n_lig = jnp.split(all_nodes, [g_rec.n_node], axis=0)

                # setting up graph with new features
                g_out = jraph.GraphsTuple(
                    nodes=n_lig, edges=e_lig,
                    senders=g_lig.senders, receivers=g_lig.receivers,
                    n_node=g_lig.n_node, n_edge=g_lig.n_edge,
                    globals=g_lig.globals)

                # aggregation
                g_out = GraphNetwork(
                    update_node_fn=tr_nodes_update,
                    update_edge_fn=tr_edges_update,
                    update_global_fn=tr_glob_update)(g_out)

                # computing T
                rot = jax.nn.softmax(rotator(g_out.globals))
                rot = rot_from_pred(rot)
                trasl = traslator(g_out.globals)
                
                # updated cloud 
                c_lig = jnp.matmul(rot, jnp.transpose(c_lig))
                c_lig = jnp.transpose(c_lig + trasl[:, None])
                all_points = jnp.concatenate([c_rec, c_lig], axis=0)
                
                # updated 2D features
                cmap = jnp.sqrt(jnp.sum(
                    (all_points[:, None, :]-all_points[None, :, :])**2, axis=-1))
                all_edges = one_hot_cmap(cmap)

                # embed 2D features
                all_edges = e_enc(all_edges)

            # update status lists
            if jnp.any(cmap<=8): 
                n_rec = up_base[idx_rec]
                n_lig = ch_base+up_base[idx_lig]
                nodes = jnp.concatenate([n_rec, n_lig], axis=0)

                senders, receivers = jnp.indices(cmap)
                edges = all_edges[senders, receivers]
                
                base_graph = jraph.GraphsTuple(
                    nodes=nodes,
                    edges=edges, 
                    senders=senders, 
                    receivers=receivers,
                    n_node=jnp.array([len(nodes)]),
                    n_edge=jnp.array([len(edges)]),
                    globals=jnp.array([1]*8))
                
                # nodes convolution
                #subg = distance_subgraph(base_graph)
                #subg = GraphConvolution(update_node_fn=neigh_nodes_update)(subg)
                #up_graph = up_graph._replace(nodes=subg.nodes)

                # graph aggregation
                up_graph = aggregation1(base_graph)
                summary = jax.nn.relu(graph_summary(up_graph.globals))

                up_lbl[idx_rec] += ch_lbl+up_lbl[idx_lig]

                # update graph, clouds and rec. choices vectors
                if idx_rec < len(choice)-1:
                    up_base = [up_base[:idx_rec], base_graph, up_base[idx_rec+1:]]
                    up_graphs = [up_graphs[:idx_rec], up_graph, up_graphs[idx_rec+1:]]
                    up_clouds = [up_clouds[:idx_rec], all_points, up_clouds[idx_rec+1:]]
                    choice = [choice[:idx_rec], summary, choice[idx_rec+1:]]
                else:
                    up_base = [up_base[:idx_rec], base_graph]
                    up_graphs = [up_graphs[:idx_rec], up_graph]
                    up_clouds = [up_clouds[:idx_rec], all_points]
                    choice = [choice[:idx_rec], summary]
 
                up_base = jnp.concatenate(up_graphs)
                up_graphs = jnp.concatenate(up_graphs)
                up_clouds = jnp.concatenate(up_clouds)
                choice = jnp.concatenate(choice)

                if idx_lig >= len(ch_graph): 
                    idx_lig -= len(ch_graph)
                    del(up_lbl[idx_lig])
                    
                    if idx_lig < len(choice)-1 and idx_lig != idx_rec:
                        up_base = [up_base[:idx_lig], up_base[idx_lig+1:]]
                        up_graphs = [up_graphs[:idx_lig], up_graphs[idx_lig+1:]]
                        up_clouds = [up_clouds[:idx_lig], up_clouds[idx_lig+1:]]
                        choice = [choice[:idx_lig], choice[idx_lig+1:]]
                        up_base = jnp.concatenate(up_graphs)
                        up_graphs = jnp.concatenate(up_graphs)
                        up_clouds = jnp.concatenate(up_clouds)
                        choice = jnp.concatenate(choice)

                    elif idx_lig != idx_rec:
                        up_base = up_base[:idx_lig]
                        up_graphs = up_graphs[:idx_lig]
                        up_clouds = up_clouds[:idx_lig]
                        choice = choice[:idx_lig]

            # batch in a single graph
            full_graph = jraph.batch(up_graphs)

            # evaluate global status
            full_graph = GraphNetwork(
                    update_node_fn=all_nodes_update,
                    update_edge_fn=all_edges_update,
                    update_global_fn=all_glob_update)(full_graph)
            incomplete = jnp.argmax(jax.nn.softmax(global_state(full_graph.globals)))

        return [up_graphs, up_lbl]
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
