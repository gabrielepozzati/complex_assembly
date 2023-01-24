import jax
import sys
import math
import haiku as hk
import jax.numpy as jnp
import jax.random as random

def get_non_lin(nl_type, negative_slope):
    assert nl_type == 'lkyrelu' or nl_type == 'swish' \
            'type must be lkyrelu or swish'

    if nl_type == 'swish': return jax.nn.SiLU()
    else: return jax.nn.LeakyReLU(negative_slope=negative_slope)



def get_layer_norm(name, ln_type, ax):
    assert ln_type=='BN' or ln_type=='LN' or ln_type == 'ID' \
            'type must be BN, LN or ID'

    if ln_type == 'BN': return hk.BatchNorm(create_scale=True, create_offset=True, name=name)
    elif ln_type == 'LN': return hk.LayerNorm(axis=ax, create_scale=True, create_offset=True, name=name)
    elif ln_type == 'GN': return GraphNorm(dim)
    else: return nn.Identity()



def apply_layer_norm(g, h, node_type, norm_type, norm_layer):
    if norm_type == 'GN': return norm_layer(g, h, node_type)
    else: return norm_layer(h)



def get_initializer(init_type, input_shape):
    assert init_type=='zeros' or init_type=='relu' or init_type=='linear' \
            'type must be zeros or relu or linear'
    
    if init_type == 'zeros': return hk.initializers.Constant(0.0)
    
    truncation_factor = jnp.asarray(.87962566103423978, 
            dtype=jnp.float32)

    scale = 1.
    for channel_dim in input_shape: scale /= channel_dim
    if init_type == 'relu': scale *= 2

    stddev = np.sqrt(scale)
    stddev = stddev / truncation_factor
    return hk.initializers.TruncatedNormal(mean=0.0, stddev=stddev)


class GraphNorm(hk.Module):
    """
        Param: []
    """
    def __init__(self, num_features, eps=1e-5, affine=True, is_node=True):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.affine = affine
        self.is_node = is_node

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(self.num_features))
            self.beta = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def norm(self, x):
        mean = x.mean(dim = 0, keepdim = True)
        var = x.std(dim = 0, keepdim = True)
        return (x - mean) / (var + self.eps)

    def forward(self, g, h, node_type):
        graph_size  = g.batch_num_nodes(node_type) if self.is_node else g.batch_num_edges(node_type)
        x_list = torch.split(h, graph_size.tolist())
        norm_list = []
        for x in x_list:
            norm_list.append(self.norm(x))
        norm_x = torch.cat(norm_list, 0)

        if self.affine:
            return self.gamma * norm_x + self.beta
        else:
            return norm_x

class Linear(hk.Module):
  
    def __init__(self,
        name,
        out_shape,
        in_dims = 1,
        initializer = 'linear',
        use_bias = True,
        bias_init = 0.):
        
        super().__init__(name=name)
        
        self.name = name
        self.in_dims = in_dims
        if type(out_shape) is int:
            self.out_dims = 1
            self.out_shape = (out_shape,)
        else:
            self.out_dims = len(out_shape)
            self.out_shape = tuple(out_shape)
        
        self.initializer = initializer
        self.use_bias = use_bias
        self.bias_init = bias_init

    def __call__(self, x):
        in_dims = self.in_dims
        if self.in_dims > 0: in_shape = x.shape[-self.in_dims:]
        else: in_shape = ()

        weight_shape = in_shape + self.out_shape
        weight_init = get_initializer(self.initializer, in_shape)
        weights = hk.get_parameter(self.name, weight_shape, x.dtype, weight_init)

        in_letters = 'abcde'[:self.in_dims]
        out_letters = 'hijkl'[:self.out_dims]
        equation = f'...{in_letters}, {in_letters}{out_letters}->...{out_letters}'

        x = jnp.einsum(equation, x, weights)

        if self.use_bias:
            bias = hk.get_parameter(name+'_bias', self.out_shape, x.dtype,
                                    hk.initializers.Constant(self.bias_init))
            x += bias

        return x


class IEGMN_Layer(hk.Module):

    def __init__(self,
                 name, n_hid_dim, n_out_dim,
                 fine_tune, config):

        super().__init__(name=name)
        
        self.cross_msgs = None
        layer_norm_coors = None
        self.final_h_layer_norm = None
        self.use_dist_in_layers = None
        
        norm_type = 'LN'
        nonlin = 'lkyrelu'
        dropout = config['actor']['dropout']
        lkyrelu_slope = config['actor']['lkyrelu_slope']
        self.c_weight = config['actor']['coord_update_weight']
        self.f_weight = config['actor']['feat_update_weight']        
        self.sigmas = [1.5 ** x for x in range(15)]
        self.n_hid_dim = n_hid_dim
        self.n_out_dim = n_out_dim
        self.fine_tune = fine_tune

        self.edge_fc1 = Linear(name+'_linear_1', n_out_dim)
        #dropout
        self.edge_act1 = get_non_lin(nonlin, lkyrelu_slope)
        self.edge_ln1 = get_layer_norm(name+'_norm_1', norm_type, n_out_dim)
        self.edge_fc2 = Linear(name+'_linear_2', n_out_dim)

        self.att_mlp_Q = hk.Sequential(
            Linear(name+'_linear_Q1', n_hid_dim, bias=False),
            get_non_lin(nonlin, lkyrelu_slope))
        
        self.att_mlp_K = hk.Sequential(
            Linear(name+'_linear_K1', n_hid_dim, bias=False),
            get_non_lin(nonlin, lkyrelu_slope))
        
        self.att_mlp_V = Linear(name+'_linear_V1', n_hid_dim, bias=False)

        self.node_ln1 = get_layer_norm(name+'_norm_2', norm_type, n_out_dim)
        self.node_fc1 = Linear(name+'_linear_3', n_hid_dim)
        #dropout
        self.node_act1 = get_non_lin(nonlin, lkyrelu_slope)
        self.node_ln2 = get_layer_norm(name+'_norm_3', norm_type, n_hid_dim)
        self.node_fc2 = Linear(name+'_linear_4', n_out_dim)
        self.node_ln3 = get_layer_norm(name+'_norm_4', norm_type, n_out_dim)

        self.coors_fc1 = Linear(name+'_linear_5', n_out_dim)
        #dropout
        self.coors_act1 = get_non_lin(nonlin, lkyrelu_slope)
        self.coors_ln1 = get_layer_norm(name+'_norm_5', norm_type, n_out_dim)
        self.coors_fc2 = Linear(name+'_linear_6', 1)

        if self.fine_tune:
            self.att_mlp_fine_Q = hk.Sequential(
                Linear(name+'_linear_Q2', n_hid_dim, bias=False),
                get_non_lin(nonlin, lkyrelu_slope))

            self.att_mlp_fine_K = hk.Sequential(
                Linear(name+'_linear_K2', n_hid_dim, bias=False),
                get_non_lin(nonlin, lkyrelu_slope))

            self.att_mlp_fine_V = hk.Sequential(
                Linear(name+'_linear_V2', n_hid_dim),
                get_non_lin(nonlin, lkyrelu_slope),
                Linear(name+'_linear_V3', 1))

    def __call__(self, key, is_training,
                 c_rec, f_rec, oc_rec, of_rec, e_rec, s_rec, r_rec, m_rec,
                 c_lig, f_lig, oc_lig, of_lig, e_lig, s_lig, r_lig, m_lig):
        '''
        prefix                          shape
        c = coordinates                 (B,N,3)
        f = node features               (B,N,ND)
        oc = original coordinates       (B,N,3)
        of = original features          (B,N,NDi)
        e = edge features               (B,N*10,EDi)
        s = sender node (j)             (B,N*10)
        r = receiver node (i)           (B,N*10)
        m = mask (0 for padding)        (B,N)

        B = batch size
        N = number of residues
        ND = dimensionality of node features
        NDi = initial dimensionality of node features
        EDi = initial dimensionality of edge features

        Edge features:
        0:3 - relative position
        3:12 - relative orientation
        12: - distance features
        '''

        def _edges_mp_input(f, c, s, r, e):
            ij_dist = vmap(lambda x, y: 
                    jnp.exp(-jnp.sum((x-y)**2)/sigma))(c[r], c[s])

            return vmap(lambda a, b, c, d:
                    jnp.concatenate((a, b, c, d), axis=-1))(f[r], f[s], ij_dist, e)

        def _cross_attention(queries, keys, values, mask):
            a = mask * jnp.matmul(queries, jnp.transpose(keys, 1, 0)) - 1000. * (1. - mask)
            a = jax.nn.softmax(a, axis=1)
            # missing weight matrix here?
            return jnp.matmul(a, values)

        def _neigh_coord_update(is_training, c, m, s, r):

            def _coors_mlp(is_training, x):
                x = self.coors_fc1(x)
                if is_training: x = hk.dropout(key1, self.dropout, x)
                x = self.coors_act1(x)
                x = self.coors_ln1(x)
                return self.coors_fc2(x)
            
            coors_mlp = partial(is_training)
            neigh_map_update = vmap(lambda x1, x2, y, z: (x1-x2)*coors_mlp(y[z]))
            updates = neigh_map_update(c[s], c[r], m, m_indices)
            updates = jnp.reshape(updates, shape=(c.shape[0], 10, 3))
            return jnp.sum(updates, axis=1)

        key1, key2, key3 = random.split(key, num=3)

        # compute m_i
        sigma = jnp.indices((15,))[0]
        grouped_shape = (m_rec.shape[0], 10, m_rec.shape[1])
        
        m_edges_mp_input = vmap(_edges_mp_input)
        empi_rec = m_edges_mp_input(f_rec, c_rec, s_rec, r_rec, e_rec)
        empi_lig = m_edges_mp_input(f_lig, c_lig, s_lig, r_lig, e_lig)

        mji_rec = self.edge_fc1(empi_rec)
        if is_training: mji_rec = hk.dropout(key1, self.dropout, mji_rec)
        mji_rec = self.edge_act1(mji_rec)
        mji_rec = self.edge_ln1(mji_rec)
        mji_rec = self.edge_fc2(mji_rec)
 
        mji_lig = self.edge_fc1(empi_lig)
        if is_training: mji_lig = hk.dropout(key1, self.dropout, mji_lig)
        mji_lig = self.edge_act1(mji_lig)
        mji_lig = self.edge_ln1(mji_lig)
        mji_lig = self.edge_fc2(mji_lig)
        
        grouped_shape = (m_rec.shape[0], 10, m_rec.shape[1])
        mji_rec_grouped = jnp.reshape(mji_rec, shape=grouped_shape)
        mji_lig_grouped = jnp.reshape(mji_lig, shape=grouped_shape)
        
        mi_rec = jnp.sum(mji_rec_grouped, axis=1)/mji_rec_grouped.shape[1]
        mi_lig = jnp.sum(mji_lig_grouped, axis=1)/mji_lig_grouped.shape[1]

        # compute mu_i
        m_cross_attention = vmap(_cross_attention)
        mask = vmap(lambda x, y: jnp.matmul(x[:, None], y[None, :]))(m_rec, m_lig)
        mui_rec = m_cross_attention(
                self.att_mlp_Q(f_rec),
                self.att_mlp_K(f_lig),
                self.att_mlp_V(f_lig), mask)
        
        mui_lig = m_cross_attention(
                self.att_mlp_Q(f_lig),
                self.att_mlp_K(f_rec),
                self.att_mlp_V(f_rec), mask.transpose(0,1))

        # compute coord updates
        m_indices = jnp.indices(s_rec.shape[-1])[0]
        c_rec_up = self.c_weight * oc_rec + (1-self.c_weight) * c_rec + \
                   _neigh_coord_update(c_rec, mji_rec, s_rec, r_rec)
        
        c_lig_up = self.c_weight * oc_rec + (1-self.c_weight) * c_rec + \
                   _neigh_coord_update(c_lig, mji_lig, s_lig, r_lig) 


        if self.fine_tune:
            f_rec_att = self.att_mlp_fine_V(f_rec)
            f_lig_att = self.att_mlp_fine_V(f_lig)
              
            c_rec_att = compute_cross_attention(
                    self.att_mlp_fine_Q(f_rec),
                    self.att_mlp_fine_K(f_lig),
                    c_lig, mask)

            c_lig_att = compute_cross_attention(
                    self.att_mlp_fine_Q(f_lig),
                    self.att_mlp_fine_K(f_rec),
                    c_rec, mask.transpose(0,1))

            c_rec_up = c_rec_up + f_rec_att * (c_rec - c_rec_att)
            c_lig_up = c_lig_up + f_lig_att * (c_lig - c_lig_att)


        f_rec_up = jnp.concatenate((self.node_norm(f_rec),
                                    mi_rec, mui_rec, of_rec), axis=-1)

        f_lig_up = jnp.concatenate((self.node_norm(f_lig),
                                    mi_lig, mui_lig, of_lig), axis=-1)

        # Skip connections
        if self.n_hid_dim == self.n_out_dim:
            f_rec_up = self.f_weight * self.node_mlp(f_rec_up) + (1.-self.f_weight) * f_rec
            f_lig_up = self.f_weight * self.node_mlp(f_lig_up) + (1.-self.f_weight) * f_lig
        else:
            f_rec_up = self.node_mlp(f_rec_up)
            f_lig_up = self.node_mlp(f_lig_up)

        # layer norm on f_rec_up and f_lig_up here?

        return c_rec_up, f_rec_up, c_lig_up, f_lig_up


class Actor(hk.Module):

    def __init__(self, name, n_lays, fine_tune, config):
        super().__init__(name=name)

        n_inp_dim = config['actor']['node_emb_dim']
        n_hid_dim = config['actor']['iegmn_hid_dim']
        lkyrelu_slope = config['actor']['lkyrelu_slope']
        heads_num = config['actor']['heads_num']
        dropout = config['actor']['dropout']

        key = random.PRNGKey(config['random_seed'])
        key1, key2, key3 = random.split(key, num=3)

        self.emb_layer = hk.Embed(vocab_size=21, embed_dim=n_inp_dim)

        self.iegmn_layers = []

        self.iegmn_layers.append(
            IEGMN_Layer(name+'_iegmn1', key1,
                        n_inp_dim,
                        n_hid_dim,
                        fine_tune, config))

        if config['actor']['shared_layers']:       
            shared_layer = IEGMN_Layer(name+'_iegmn2', key2,
                                       n_hid_dim,
                                       n_hid_dim,
                                       fine_tune, config)
            for layer_idx in range(1, n_lays):
                self.iegmn_layers.append(shared_layer)

        else:
            for layer_idx in range(1, n_lays):
                iegmn_name = name+'_iegmn'+str(layer_idx+1)
                self.iegmn_layers.append(
                    IEGMN_Layer(iegmn_name, key3,
                                n_hid_dim,
                                n_hid_dim,
                                fine_tune, config))

        self.att_mlp_query_ROT = Linear(name+'_linear_Q', heads_num*n_hid_dim, bias=False)

        self.att_mlp_key_ROT = Linear(name+'_linear_K', heads_num*n_hid_dim, bias=False)

        self.mlp_h_mean_ROT = hk.Sequential(
            Linear(name+'_Linear_MLP', n_hid_dim),
            hk.Dropout(dropout),
            get_non_lin('lkyrelu', lkyrelu_slope))


    def __call__(self, 
                 c_rec, f_rec, e_rec, s_rec, r_rec, m_rec,
                 c_lig, f_lig, e_lig, s_lig, r_lig, m_lig):

        def centroid_attention(feat, feat_mean, coord):
            key = jnp.reshape(self.att_mlp_key_ROT(feat), (-1, self.num_att_heads, d))
            key = key.transpose(0,1,2)                                                                          # (H,N,d)

            query = jnp.reshape(self.att_mlp_query_ROT(feat_mean), (1, self.num_att_heads, d))
            query = query.transpose(0,2,1)                                                                      # (H,d,N)

            att_vec = jnp.reshape(jax.nn.softmax(key @ (query/jnp.sqrt(d)), dim=1), (self.num_att_heads, -1))
            att_vec = att_vec @ coord
            return att_vec

        def kabsch(Y_rec, Y_lig):
            Y_rec_mean = jnp.mean(Y_rec, dim=0, keepdim=True)
            Y_lig_mean = jnp.mean(Y_lig, dim=0, keepdim=True)

            A = jnp.transpose(Y_rec - Y_rec_mean) @ (Y_lig - Y_lig_mean)
            U, S, Vt = jnp.linalg.svd(A)

            #num_it = 0
            #while jnp.min(S) < 1e-3 or jnp.min(jnp.abs(jnp.reshape((S**2), (1,3)) - jnp.reshape((S**2), (3,1)) + jnp.eye(3))) < 1e-2:

            #    A = A + jax.random.uniform(key, (3,3)) * jnp.eye(3)
            #    U, S, Vt = jnp.linalg.svd(A)
            #    num_it += 1

                #if num_it > 10:
                #    print ('SVD consistently numerically unstable! Exitting ... ')
                #    sys.exit(1)

            corr_mat = jnp.diag(jnp.array([1,1,jnp.sign(jnp.linalg.det(A))]))
            T_align = (U @  corr_mat) @ Vt

            b_align = Y_rec_mean - jnp.transpose(T_align @ jnp.transpose(Y_lig_mean))
            return T_align, b_align

        f_rec = self.emb_layer(f_rec)
        f_lig = self.emb_layer(f_lig)

        of_rec, of_lig = f_rec, f_lig
        oc_rec, oc_lig = c_rec, c_lig

        for i, layer in enumerate(self.iegmn_layers):
            
            c_rec, f_rec, c_lig, f_lig = layer(
                    c_rec, f_rec, oc_rec, of_rec,
                    e_rec, s_rec, r_rec, m_rec,
                    c_lig, f_lig, oc_lig, of_lig, 
                    e_lig, s_lig, r_lig, m_lig)

        d = f_rec.shape[-1]
        f_rec_mean = jnp.mean(self.mlp_h_mean_ROT(f_rec), dim=1, keepdim=True)
        f_lig_mean = jnp.mean(self.mlp_h_mean_ROT(f_lig), dim=1, keepdim=True)

        Y_rec = vmap(centroid_attention)(f_rec, f_lig_mean, c_rec)
        Y_lig = vmap(centroid_attention)(f_lig, f_rec_mean, c_lig)

        all_T_align, all_b_align = vmap(kabsch)(Y_rec, Y_lig)

        return jnp.concatenate((all_T_align, all_b_align), axis=-1)


class Critic(hk.Module):
    def __init__(self, name, n_lays, config):
        super().__init__(name=name)

        self.actor_submodule = Actor(name, n_lays, True, config)

        # final score estimation attention / coordinates

        self.att_score_coord_Q = hk.Sequential(
                Linear(name+'_linear_Qc', heads_num*n_hid_dim, bias=False),
                get_non_lin(nonlin, lkyrelu_slope))

        self.att_score_coord_K = hk.Sequential(
                Linear(name+'_linear_Kc', heads_num*n_hid_dim, bias=False),
                get_non_lin(nonlin, lkyrelu_slope))

        self.att_score_coord_V = hk.Sequential(
                Linear(name+'_linear_Vc', heads_num*n_hid_dim),
                get_non_lin(nonlin, lkyrelu_slope))

        self.att_score_coord = hk.Sequential(
            Linear(name+'_linear_2', n_hid_dim, in_dims=2),
            hk.Dropout(dropout),
            get_non_lin(nonlin, lkyrelu_slope),
            get_layer_norm(name+'_norm_1', norm_type, n_out_dim))

        # final score estimation attention / features

        self.att_score_feat_Q = hk.Sequential(
                Linear(name+'_linear_Qf', heads_num*n_hid_dim, bias=False),
                get_non_lin(nonlin, lkyrelu_slope))

        self.att_score_feat_K = hk.Sequential(
                Linear(name+'_linear_Kf', heads_num*n_hid_dim, bias=False),
                get_non_lin(nonlin, lkyrelu_slope))

        self.att_score_feat_V = hk.Sequential(
                Linear(name+'_linear_Vf', heads_num*n_hid_dim),
                get_non_lin(nonlin, lkyrelu_slope))

        self.att_score_feat = hk.Sequential(
            Linear(name+'_linear_3', n_hid_dim, in_dims=2),
            hk.Dropout(dropout),
            get_non_lin(nonlin, lkyrelu_slope),
            get_layer_norm(name+'_norm_2', norm_type, n_hid_dim))

    def __call__(self, actions,
                 c_rec, f_rec, e_rec, s_rec, r_rec, m_rec,
                 c_lig, f_lig, e_lig, s_lig, r_lig, m_lig):

        def get_next_coord(c, action):
            rot = jnp.reshape(action[:9], shape=(3,3))
            trasl = jnp.reshape(action[9:], shape=(3))
            return ( rot @ c.transpose(1,0) ).transpose(1,0) + trasl

        def cross_attention(c_lig, c_rec, f_lig, f_rec):
            shape = (self.pad_len, self.heads_num, self.n_hid_dim)          # (N,H,d)
            
            c_q = jnp.reshape(self.att_score_coord_Q(c_lig), shape=shape)
            c_k = jnp.reshape(self.att_score_coord_K(c_rec), shape=shape)
            c_v = jnp.reshape(self.att_score_coord_V(c_rec), shape=shape)
            
            f_q = jnp.reshape(self.att_score_feat_Q(f_lig), shape=shape)
            f_k = jnp.reshape(self.att_score_feat_K(f_rec), shape=shape)
            f_v = jnp.reshape(self.att_score_feat_V(f_rec), shape=shape) 
            
            affine_c = c_q.transpose(1,0,2) @ c_k.transpose(1,2,0)          # (H,Nl,d)@(H,d,Nr)
            affine_f = f_q.transpose(1,0,2) @ f_k.transpose(1,2,0)
            affine = jax.nn.softmax(affine_c + affine_f, axes=-1)           # (H,Nl,Nr)

            c_out = (affine @ c_v.transpose(1,0,2)).transpose(1,0,2)        # (H,Nl,Nr)@(H,Nr,d) = (H,Nl,d) --> (Nl,H,d)
            f_out = (affine @ f_v.transpose(1,0,2)).transpose(1,0,2)
            return self.att_score_coord(c_out) + self.att_score_feat(f_out)

        c_lig1 = c_lig
        c_lig2 = vmap(get_next_coord)(c_lig, actions)

        actions1 = self.actor_submodule(
                 c_rec, f_rec, e_rec, s_rec, r_rec, m_rec,
                 c_lig1, f_lig, e_lig, s_lig, r_lig, m_lig)

        actions2 = self.actor_submodule(
                 c_rec, f_rec, e_rec, s_rec, r_rec, m_rec,
                 c_lig2, f_lig, e_lig, s_lig, r_lig, m_lig)

        c_lig2P = vmap(get_next_coord)(c_lig1, actions1)
        c_lig3 = vmap(get_next_coord)(c_lig2, actions2)

        scores1 = vmap(cross_attention)(c_lig2P, c_rec, f_lig, f_rec) 
        scores2 = vmap(cross_attention)(c_lig3, c_rec, f_lig, f_rec)
        return jnp.sum((scores1,scores2), axis=0)
         
