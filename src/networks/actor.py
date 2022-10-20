import haiku as hk
from networks.modules import *

class Actor(hk.Module):
 
    def __init__(self, num_heads, num_channels):
        super().__init__(name='Actor')

        #base node/edge encoding
        self.encoding = GraphEncoding(num_channels)

        # surface encoding
        self.single_stack = [AttentionGraphStack(num_heads, num_channels, edge_a=False) \
                for n in range(4)] 

        self.interaction_stack = [AttentionGraphStack(num_heads, num_channels) \
                for n in range(4)]

        # pair encoding
        self.pair_stack = [
                [AttentionGraphStack(num_heads, num_channels, edge_a=False),
                 AttentionGraphStack(num_heads, num_channels, edge_a=False)] \
                for n in range(8)]

        # transformation graph updates
        self.docking_stack = [AttentionGraphStack(num_heads, num_channels, edge_a=False) \
                for n in range(8)] 

        # roto-traslation output
        self.out_rblock = MultiLayerPerceptron(num_channels, num_channels, 1)
        self.out_tblock = MultiLayerPerceptron(num_channels, num_channels, 1)
        self.traslator = Linear(3)
        self.rotator = Linear(3)

        # confidence output
        self.out_cblock = MultiLayerPerceptron(num_channels, num_channels, 1)
        self.confidence = Linear(1)

    def __call__(self, g_rec, g_lig, g_int):
        # receptor/ligand encoding
        g_rec = self.encoding(g_rec)
        g_lig = self.encoding(g_lig)

        # elaborate surface level info
        for module in self.single_stack:
            g_rec = module(g_rec)
            g_lig = module(g_lig)

        # elaborate interface level info
        g_int = g_int._replace(
                nodes=jnp.concatenate((g_rec.nodes, g_lig.nodes), axis=0),
                edges=self.encoding.e_enc(g_int.edges))
        for module in self.interaction_stack: g_int = module(g_int)

        # elaborate mutual surface info
        for module1, module2 in self.pair_stack:
            g_rec = module1(g_rec, g_lig)
            g_lig = module2(g_lig, g_rec)

        # union of rec, lig and int edges and skip connection of int nodes  
        all_nodes = jnp.concatenate((g_rec.nodes, g_lig.nodes), axis=0)
        all_edges = jnp.concatenate(
                (g_rec.edges, g_lig.edges, g_int.edges), axis=0)
        all_senders = jnp.concatenate(
                (g_rec.senders, g_lig.senders, g_int.senders), axis=0)
        all_receivers = jnp.concatenate(
                (g_rec.receivers, g_lig.receivers, g_int.receivers), axis=0)
        agg_globals = g_rec.globals+g_lig.globals

        g_int = jraph.GraphsTuple(
                nodes=all_nodes+g_int.nodes, edges=all_edges,
                senders=all_senders, receivers=all_receivers,
                n_node=jnp.array([all_nodes.shape[0]]), 
                n_edge=jnp.array([all_edges.shape[0]]),
                globals=agg_globals)

        # elaborate all info to derive final global feature
        for module in self.docking_stack: g_int = module(g_int)
        
        # elaborate final global feature to rot. and trasl.
        out_r = self.out_rblock(g_int.globals)
        out_t = self.out_tblock(g_int.globals+out_r)
        out_t = self.traslator(out_t)
        out_r = jnp.tanh(self.rotator(out_r))

        # elaborate confidence estimation
        out_c = self.out_cblock(g_int.globals)
        out_c = jax.nn.sigmoid(self.confidence(out_c))

        return jnp.concatenate((out_r, out_t, out_c), axis=-1)

