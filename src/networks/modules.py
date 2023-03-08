
import numbers
import functools
import haiku as hk
import numpy as np

import jax
import jax.numpy as jnp
import jax.tree_util as tree
from jax import grad, jit, vmap
from jax import random

from ops import *

from typing import Any, Union, Sequence, Mapping, Optional

TRUNCATED_NORMAL_STDDEV_FACTOR = np.asarray(.87962566103423978,
                                            dtype=np.float32)

def get_initializer_scale(initializer_name, input_shape):
  """Get Initializer for weights and scale to multiply activations by."""

  if initializer_name == 'zeros':
    w_init = hk.initializers.Constant(0.0)
  else:
    # fan-in scaling
    scale = 1.
    for channel_dim in input_shape:
      scale /= channel_dim
    if initializer_name == 'relu':
      scale *= 2

    noise_scale = scale

    stddev = np.sqrt(noise_scale)
    # Adjust stddev for truncation.
    stddev = stddev / TRUNCATED_NORMAL_STDDEV_FACTOR
    w_init = hk.initializers.TruncatedNormal(mean=0.0, stddev=stddev)
  
  return w_init


def traslate_point(point, traslation, extra_dims=1):
  #for _ in range(extra_dims):
  #    expand_fn = functools.partial(jnp.expand_dims, axis=-1)
  #    traslation = jax.tree_map(expand_fn, traslation)
  traslation = jnp.transpose(traslation)

  return [point[0] + jnp.expand_dims(traslation[0], axis=-1),
          point[1] + jnp.expand_dims(traslation[1], axis=-1),
          point[2] + jnp.expand_dims(traslation[2], axis=-1)]

def squared_difference(x, y):
  return jnp.square(x - y)

class Linear(hk.Module):
  """Protein folding specific Linear module.
  This differs from the standard Haiku Linear in a few ways:
    * It supports inputs and outputs of arbitrary rank
    * Initializers are specified by strings
  """

  def __init__(self,
      num_output: Union[int, Sequence[int]],
      initializer: str = 'linear',
      num_input_dims: int = 1,
      use_bias: bool = True,
      bias_init: float = 0.,
      precision = None,
      name: str = 'linear'):
    """Constructs Linear Module.
    Args:
      num_output: Number of output channels. Can be tuple when outputting
          multiple dimensions.
      initializer: What initializer to use, should be one of {'linear', 'relu',
        'zeros'}
      num_input_dims: Number of dimensions from the end to project.
      use_bias: Whether to include trainable bias
      bias_init: Value used to initialize bias.
      precision: What precision to use for matrix multiplication, defaults
        to None.
      name: Name of module, used for name scopes.
    """
    super().__init__(name=name)
    if isinstance(num_output, numbers.Integral):
      self.output_shape = (num_output,)
    else:
      self.output_shape = tuple(num_output)
    self.initializer = initializer
    self.use_bias = use_bias
    self.bias_init = bias_init
    self.num_input_dims = num_input_dims
    self.num_output_dims = len(self.output_shape)
    self.precision = precision

  def __call__(self, inputs):
    """Connects Module.
    Args:
      inputs: Tensor with at least num_input_dims dimensions.
    Returns:
      output of shape [...] + num_output.
    """
    num_input_dims = self.num_input_dims

    if self.num_input_dims > 0:
      in_shape = inputs.shape[-self.num_input_dims:]
    else:
      in_shape = ()

    weight_init = get_initializer_scale(self.initializer, in_shape)
    weight_shape = in_shape + self.output_shape
    weights = hk.get_parameter('weights', weight_shape, inputs.dtype,
                               weight_init)

    in_letters = 'abcde'[:self.num_input_dims]
    out_letters = 'hijkl'[:self.num_output_dims]
    equation = f'...{in_letters}, {in_letters}{out_letters}->...{out_letters}'

    output = jnp.einsum(equation, inputs, weights, precision=self.precision)

    if self.use_bias:
      bias = hk.get_parameter('bias', self.output_shape, inputs.dtype,
                              hk.initializers.Constant(self.bias_init))
      output += bias

    return output


class MLP(hk.Module):

  def __init__(self, num_channels, out_channels, num_layers, name):

    super().__init__(name=name)
    
    self.stack = [Linear(num_channels, name=name+f'_linear{n}') \
            for n in range(num_layers)]

    self.out_layer = Linear(out_channels, name=name+'_linear_last')

  def __call__(self, x):

    for layer in self.stack:
      x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
      x = jax.nn.relu(layer(x))

    x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
    
    return self.out_layer(x)


class MHA(hk.Module):
  """Multi-headed attention mechanism.
     As described in the vanilla Transformer paper:
    "Attention is all you need" https://arxiv.org/abs/1706.03762
  """

  def __init__(
      self,
      num_heads: int,
      key_size: int,
      #w_init_scale: float,
      value_size: Optional[int] = None,
      model_size: Optional[int] = None,
      name: Optional[str] = None):

    super().__init__(name=name)
    self.num_heads = num_heads
    self.key_size = key_size
    self.value_size = value_size or key_size
    self.model_size = model_size or key_size
    #self.w_init = hk.initializers.VarianceScaling(w_init_scale)

  def __call__(
      self,
      query: jnp.ndarray,
      key: jnp.ndarray,
      value: jnp.ndarray) -> jnp.ndarray:

    query_heads = self._linear_projection(query, self.key_size)
    key_heads = self._linear_projection(key, self.key_size)
    value_heads = self._linear_projection(value, self.value_size)

    attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
    attn_logits = attn_logits / jnp.sqrt(self.key_size).astype(key.dtype)

    attn_weights = jax.nn.softmax(attn_logits)
    attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
    
    # Concatenate attention matrix of all heads into a single vector.
    attn_vec = jnp.reshape(attn, (*query.shape[:-1], -1))

    return hk.Linear(self.model_size)(attn_vec)

  @hk.transparent
  def _linear_projection(
      self,
      x: jnp.ndarray,
      head_size: int,
      name: Optional[str] = None) -> jnp.ndarray:

    y = Linear(self.num_heads * head_size)(x)
    return y.reshape((*x.shape[:-1], self.num_heads, head_size))


class MPNN(hk.Module):
  '''
  Neural message passing for quantum chemistry. 
  J.Gilmer et al 2017
  '''

  def __init__(self, config, edge_up, name='MPNN'):

    super().__init__(name=name)

    channels = config['channels']
    layer_num = config['mlp_layers']
    self.edge_num = config['edge_number']
    self.edge_up = edge_up

    self.message_mlp = MLP(
            channels, channels, layer_num, name=name+'_message')
    
    self.ln1 = hk.LayerNorm(axis=-1, name=name+'_ln1',
            create_scale=True, create_offset=True)
    self.node_transf1 = Linear(channels, name=name+'_nodetr1')
    self.ln2 = hk.LayerNorm(axis=-1, name=name+'_ln2',
            create_scale=True, create_offset=True)
    self.node_update = Linear(channels, name=name+'_nodeup')
    
    if self.edge_up:
      self.ln3 = hk.LayerNorm(axis=-1, name=name+'_ln3',
              create_scale=True, create_offset=True)
      self.node_transf2 = Linear(channels, name=name+'_nodetr2')
      self.ln4 = hk.LayerNorm(axis=-1, name=name+'_ln4',
              create_scale=True, create_offset=True)
      self.edge_update = Linear(channels, name=name+'_edgeup')

  def __call__(self, nodes, edges, i, j):
    n_i, n_j = nodes[i], nodes[j]
    n_ij = jnp.concatenate((n_i, n_j, edges), axis=-1)
    m_i = jax.nn.relu(self.message_mlp(n_ij))

    shape = (nodes.shape[0], 
             int(edges.shape[0]/nodes.shape[0]), 
             m_i.shape[-1])
    m_i = jnp.reshape(m_i, shape)
    m_i = jnp.sum(m_i, axis=1)
    
    nodes = self.ln1(nodes)
    n_up = jax.nn.relu(self.node_transf1(nodes))
    n_up = jnp.concatenate((n_up, m_i), axis=-1)
    n_up = self.ln2(n_up)
    n_up = jax.nn.relu(self.node_update(n_up))

    if self.edge_up:
      n_ij = jnp.concatenate((n_i, n_j), axis=-1)
      n_ij = self.ln3(n_ij)
      n_ij = jax.nn.relu(self.node_transf2(n_ij))

      e_up = jnp.concatenate((edges, n_ij), axis=-1)
      e_up = self.ln4(e_up)
      e_up = jax.nn.relu(self.edge_update(e_up))
      return n_up, e_up
    
    else: return n_up

#class GraphAttention(hk.Module):
#  '''
#  Graph attention network. 
#  P.Velickovic et al. ICLR 2018
#  '''
#
#  def __init__(self, config, name='GAT'):
#
#    super().__init__(name=name)
