import jax
import sys
import math

#import dgl
#import torch
#from torch import nn
#from dgl import function as fn
from src.utils.graph_norm import GraphNorm

import haiku as hk


def get_non_lin(nl_type, negative_slope):
    assert nl_type == 'lkyrelu' or nl_type == 'swish',
            'type must be lkyrelu or swish'

    if nl_type == 'swish': return jax.nn.SiLU()
    else: return jax.nn.LeakyReLU(negative_slope=negative_slope)



def get_layer_norm(ln_type, ax):
    assert ln_type=='BN' or ln_type=='LN' or ln_type == 'ID',
            'type must be BN, LN or ID'

    if ln_type == 'BN': return hk.BatchNorm(create_scale=True, create_offset=True)
    elif ln_type == 'LN': return hk.LayerNorm(axis=ax, create_scale=True, create_offset=True)
    elif layer_norm_type == 'GN': return GraphNorm(dim)
    else: return nn.Identity()



def apply_layer_norm(g, h, node_type, norm_type, norm_layer):
    if norm_type == 'GN': return norm_layer(g, h, node_type)
    else: return norm_layer(h)



def get_initializer(init_type, input_shape):
    assert init_type=='zeros' or init_type=='relu' or init_type=='linear',
            'type must be zeros or relu or linear'
    
    truncation_factor = jnp.asarray()
    if init_type == 'zeros': return hk.initializers.Constant(0.0)
  
    scale = 1.
    for channel_dim in input_shape: scale /= channel_dim
    if init_type == 'relu': scale *= 2

    stddev = np.sqrt(scale)
    stddev = stddev / TRUNCATED_NORMAL_STDDEV_FACTOR
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
