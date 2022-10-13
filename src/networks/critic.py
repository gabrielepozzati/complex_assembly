def Critic(num_heads: int, num_channels: int):
    """Create the model's forward pass."""

    def forward_fn(graphs, action) -> jnp.ndarray:
        """Forward pass."""
        ################################
        ##### layers/modules definitions
        #base node/edge encoding
        encoding = GraphEncoding(num_channels)
        action_encoding = MultiLayerPerceptron(num_channels, num_channels, 2)

        # surface encoding 
        single_stack = [AttentionGraphStack(num_heads, num_channels) \
                for n in range(4)]

        interaction_stack = [AttentionGraphStack(num_heads, num_channels) \
                for n in range(4)]

        # transformation graph updates
        docking_stack = [AttentionGraphStack(num_heads, num_channels) \
                for n in range(8)]

        # confidence output
        out_block = MultiLayerPerceptron(num_channels, num_channels, 1)
        value = Linear(1)

        ####### START MODEL #######
        g_rec, g_lig, g_int = graphs
        action = action_encoding(action)

        # receptor/ligand encoding
        g_rec = encoding(g_rec)
        g_lig = encoding(g_lig)
        g_rec.replace(globals=jnp.zeros(action.shape))
        g_lig.replace(globals=action)

        # elaborate surface level info
        for module in single_stack:
            g_rec = module(g_rec)
            g_lig = module(g_lig)

        # elaborate interface level info
        g_int = g_int.replace(
                nodes=jnp.concatenate((g_rec.nodes, g_lig.nodes), axis=0),
                edges=encoding.e_enc(g_int.edges),
                globals=action)

        for module in interaction_stack: g_int = module(g_int)

        # union of rec, lig and int edges and skip connection of int nodes  
        all_nodes = jnp.concatenate((g_rec.nodes, g_lig.nodes), axis=0)
        all_edges = jnp.concatenate(
                (g_rec.edges, g_lig.edges, g_int.edges), axis=0)
        all_senders = jnp.concatenate(
                (g_rec.senders, g_lig.senders, g_int.senders), axis=0)
        all_receivers = jnp.concatenate(
                (g_rec.receivers, g_lig.receivers, g_int.receivers), axis=0)
        agg_globals = jnp.sum((g_rec.globals,g_lig.globals,g_int.globals) axis=0)

        g_int = jraph.GraphsTuple(
                nodes=all_nodes+g_int.nodes, edges=all_edges,
                senders=all_senders, receivers=all_receivers,
                n_node=all_nodes.shape[0], n_edge=all_edges.shape[0],
                globals=agg_globals)

        # elaborate all info to derive final global feature
        for module in docking_stack: g_int = module(g_int)
        
        # elaborate q-value
        q = out_block(g_int.globals)
        q = value(q)

        return q
    return forward_fn

