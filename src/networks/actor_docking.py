def build_forward_fn(num_heads: int, num_channels: int):
    """Create the model's forward pass."""

    def forward_fn(clouds, nodes, edges, masks) -> jnp.ndarray:
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

