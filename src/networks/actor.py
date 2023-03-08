import haiku as hk
from networks.modules import *

class Actor(hk.Module):
 
    def __init__(self, config):
        super().__init__(name='Actor')

        self.ch = config['channels']
        self.un = config['update_number']
        self.mlp = config['mlp_layers']

        self.node_input = vmap(Linear(self.ch, name='feat_in'))
        
        self.edge_input = vmap(Linear(self.ch, name='edge_in'))

        self.node_layers1, self.node_layers2, self.edge_layers = [], [], []
        for n in range(self.un):
            self.node_layers1.append(
                    vmap(MPNN(config, False, name=f'intra_MPNN{n}')))

            self.edge_layers.append(
                    vmap(MPNN(config, True, name=f'inter_MPNN{n}')))

            self.node_layers2.append(
                    vmap(MPNN(config, False, name=f'intra_MPNN{n}')))

        self.P_join = vmap(
                MLP(self.ch, self.ch, self.mlp, name='P_join'))

        self.p1_join = vmap(
                MLP(self.ch, self.ch, self.mlp, name='p1_join'))

        self.p2_join = vmap(
                MLP(self.ch, self.ch, self.mlp, name='p2_join'))

        self.Pp1_merge = vmap(
                MLP(self.ch, self.ch, self.mlp, name='Pp1_merge'))

        self.P_out = vmap(
                MLP(int(self.ch/2), 1, self.mlp, name='P_out'))

        self.p1_out = vmap(
                MLP(int(self.ch/2), 1, self.mlp, name='p1_out'))

        self.p2_out = vmap(
                MLP(int(self.ch/2), 1, self.mlp, name='p2_out'))


    def __call__(self, masks, nodes, edges, i_s, j_s):

        (padmask_recs, padmask_ligs,
         intmask_recs, intmask_ligs, 
         rimmask_recs, rimmask_ligs, 
         node_recs, node_ligs, 
         edge_recs, edge_ligs, edge_ints,
         i_recs, i_ligs, i_ints, 
         j_recs, j_ligs, j_ints) = \
                 vmap(unfold_features)(masks, nodes, edges, i_s, j_s)

        # expand node features
        node_recs = jax.nn.relu(self.node_input(node_recs))
        node_ligs = jax.nn.relu(self.node_input(node_ligs))

        # placeholders for P, p1 and p2 pseudo-feats
        #node_Ps = jnp.zeros(node_recs.shape)
        #node_p1s = jnp.zeros(node_recs.shape)
        #node_p2s = jnp.zeros(node_recs.shape)

        # merge pseudo-feats to node features
        #node_Pp1s = jnp.concatenate((node_Ps, node_p1s), axis=2)
        #node_Pp1s = jax.nn.relu(self.Pp1_merge(node_Pp1s))

        #node_recs = jnp.concatenate((node_recs, node_p2s), axis=2)
        #node_ligs = jnp.concatenate((node_ligs, node_Pp1s), axis=2)            

        # node-wise only, intra protein MPNN
        for n in range(self.un):
            node_recs = self.node_layers1[n](node_recs,edge_recs,i_recs,j_recs)
            node_ligs = self.node_layers1[n](node_ligs,edge_ligs,i_ligs,j_ligs)
            
        # node and edge-wise, inter protein MPNN
        node_ints = jnp.concatenate((node_recs, node_ligs), axis=1)

        for n in range(self.un):
            node_ints, edge_ints = self.edge_layers[n](node_ints,edge_ints,i_ints,j_ints)

        node_recs, node_ligs = jnp.split(node_ints, 2, axis=1)

        # create P, p1 and p2 pseudofeats
        node_Ps = self.P_join(node_ligs)*padmask_ligs[:,:,None]
        node_p1s = self.p1_join(node_ligs)*padmask_ligs[:,:,None]
        node_p2s = self.p2_join(node_recs)*padmask_recs[:,:,None]
        
        # merge pseudo-feats to node features
        node_Pp1s = jnp.concatenate((node_Ps, node_p1s), axis=2)
        node_Pp1s = jax.nn.relu(self.Pp1_merge(node_Pp1s))

        node_recs = jnp.concatenate((node_recs, node_p2s), axis=2)
        node_ligs = jnp.concatenate((node_ligs, node_Pp1s), axis=2)

        # node-wise only, intra protein MPNN
        for n in range(self.un):
            node_recs = self.node_layers2[n](node_recs,edge_recs,i_recs,j_recs)
            node_ligs = self.node_layers2[n](node_ligs,edge_ligs,i_ligs,j_ligs)

        # create P, p1 and p2 output feats
        node_Ps = jax.nn.softmax(jnp.squeeze(self.P_out(node_Ps)))*intmask_ligs
        node_p1s = jax.nn.softmax(jnp.squeeze(self.p1_out(node_p1s)))*rimmask_ligs
        node_p2s = jax.nn.softmax(jnp.squeeze(self.p2_out(node_p2s)))*rimmask_recs
        
        node_Ps = jnp.squeeze(node_Ps)
        node_p1s = jnp.squeeze(node_p1s)
        node_p2s = jnp.squeeze(node_p2s)

        return jnp.stack((node_Ps,node_p1s,node_p2s), axis=1)

