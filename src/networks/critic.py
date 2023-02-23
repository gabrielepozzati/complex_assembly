import haiku as hk
from networks.modules import *

class Critic(hk.Module):
    def __init__(self, config):
        super().__init__(name='Critic')

        self.ch = config['channels']
        self.un = config['update_number']
        self.mlp = config['mlp_layers']

        self.action_update = vmap(MLP(self.ch, int(self.ch/2), self.mlp, name='action_update'))

        self.node_layers1, self.node_layers2, self.edge_layers = [], [], []
        for n in range(self.un):
            self.node_layers1.append(vmap(MPNN(config, False, name=f'action_MPNN{n}')))
            self.edge_layers.append(vmap(MPNN(config, True, name=f'pivot_MPNN{n}')))
            self.node_layers2.append(vmap(MPNN(config, False, name=f'eval_MPNN{n}')))

        self.global_update = vmap(MLP(self.ch, int(self.ch/2), self.mlp, name='global_update'))
        self.out_update = Linear(self.ch, num_input_dims=2, name='out')


    def __call__(self, masks, nodes, edges, i_s, j_s, actions):
 
        (padmask_recs, padmask_ligs, 
         intmask_recs, intmask_ligs, 
         rimmask_recs, rimmask_ligs, 
         node_recs, node_ligs, 
         edge_recs, edge_ligs, edge_ints, 
         i_recs, i_ligs, i_ints, 
         j_recs, j_ligs, j_ints) = vmap(unfold_features)(masks, nodes, edges, i_s, j_s)

        # merge a complete graph without node information
        blanks = jnp.concatenate((padmask_recs[:,:,None], padmask_ligs[:,:,None]), axis=1)
        all_edges = jnp.concatenate((edge_recs, edge_ligs, edge_ints), axis=1)
        all_is = jnp.concatenate((i_recs, i_ligs+400, i_ints), axis=1)
        all_js = jnp.concatenate((j_recs, j_ligs+400, j_ints), axis=1)

        # enrich nodes with full-graph structural info
        all_nodes = blanks
        for n in range(self.un):
            all_nodes = self.node_layers1[n](all_nodes, edges, all_is, all_js)
            all_nodes *= blanks

        # refine action feature to concatenate with nodes
        fillers = jnp.zeros(actions[:,0,:].shape)
        Ps = jnp.squeeze(actions[:,0,:])
        p1s = jnp.squeeze(actions[:,1,:])
        p2s = jnp.squeeze(actions[:,2,:])
        action_Ps = jnp.concatenate((fillers, Ps), axis=1)
        action_p1p2s = jnp.concatenate((p2s, p1s), axis=1)
        actions = jnp.concatenate(
                (action_Ps[:,:,None], action_p1p2s[:,:,None]), axis=2)

        # merge action with edge-enriched information and update
        actions = jnp.concatenate((all_nodes, actions), axis=2)
        actions = self.action_update(actions)

        # merge back into structural enriched nodes
        all_nodes = jnp.concatenate((all_nodes, actions), axis=2)

        # update edges based on full interaction graph
        for n in range(self.un):
            all_nodes, all_edges = self.edge_layers[n](all_nodes, all_edges, all_is, all_js)
            all_nodes = jnp.concatenate((all_nodes, actions), axis=2)

        # group original node info and update full interaction graph
        ori_nodes = jnp.concatenate((node_recs, node_ligs), axis=1)
        
        all_nodes = ori_nodes
        for n in range(self.un):
            all_nodes = self.node_layers2[n](all_nodes, all_edges, all_is, all_js)
            all_nodes = jnp.concatenate((all_nodes, ori_nodes), axis=2)

        # aggregate graph to a single value

        all_nodes = self.global_update(all_nodes)
        return jax.nn.sigmoid(self.out_update(all_nodes))

