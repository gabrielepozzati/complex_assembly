def actor(num_heads: int, num_channels: int):
    """Create the model's forward pass."""

    def forward_fn(graphs) -> jnp.ndarray:
        """Forward pass."""
        ################################
        ##### layers/modules definitions
        #base node/edge encoding
        encoding = GraphEncoding(num_channels)

        # surface encoding 
        single_stack = [AttentionGraphStack(num_heads, num_channels) \
                for n in range(4)]

        interaction_stack = [AttentionGraphStack(num_heads, num_channels) \
                for n in range(4)]

        # pair encoding
        pair_stack = [[AttentionGraphStack(num_heads, num_channels),
                       AttentionGraphStack(num_heads, num_channels)] \
                for n in range(4)]

        # transformation graph updates
        docking_stack = [AttentionGraphStack(num_heads, num_channels) \
                for n in range(8)]

        # roto-traslation output
        out_rblock = MultiLayerPerceptron(num_channels, num_channels, 1)
        out_tblock = MultiLayerPerceptron(num_channels, num_channels, 1)
        traslator = Linear(3)
        rotator = Linear(3)

        # confidence output
        out_cblock = MultiLayerPerceptron(num_channels, num_channels, 1)
        confidence = Linear(1)

        ####### START MODEL #######
        g_rec, g_lig, g_int = graphs
        
        # receptor/ligand encoding
        g_rec = encoding(g_rec)
        g_lig = encoding(g_lig)

        # elaborate surface level info
        for module in single_stack:
            g_rec = module(g_rec)
            g_lig = module(g_lig)

        # elaborate interface level info
        g_int = g_int.replace(
                nodes=jnp.concatenate((g_rec.nodes, g_lig.nodes), axis=0),
                edges=encoding.e_enc(g_int.edges))
        for module in interaction_stack: g_int = module(g_int)

        # elaborate mutual surface info
        for module1, module2 in pair_stack:
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
        agg_globals = jnp.sum((g_rec.globals,g_lig.globals) axis=0)

        g_int = jraph.GraphsTuple(
                nodes=all_nodes+g_int.nodes, edges=all_edges,
                senders=all_senders, receivers=all_receivers,
                n_node=all_nodes.shape[0], n_edge=all_edges.shape[0],
                globals=agg_globals)

        # elaborate all info to derive final global feature
        for module in docking_stack: g_int = module(g_int)
        
        # elaborate final global feature to rot. and trasl.
        out_r = out_rblock(g_int.globals)
        out_t = out_tblock(g_int.globals+out_r)
        out_t = traslator(out_t)
        out_r = jnp.tanh(rotator(out_r))

#        # elaborate confidence estimation
#        out_c = out_cblock(g_int.globals)
#        out_c = jax.sigmoid(confidence(out_c))

        return jnp.concatenate((out_r, out_t), axis=-1)
    return forward_fn

