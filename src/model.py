
import numbers
import functools
import haiku as hk
import numpy as np

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

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


class MultiHeadAttention(hk.Module):
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
    self.model_size = model_size or key_size * num_heads
    #self.w_init = hk.initializers.VarianceScaling(w_init_scale)

  def __call__(
      self,
      query: jnp.ndarray,
      key: jnp.ndarray,
      value: jnp.ndarray) -> jnp.ndarray:

    """Compute (optionally masked) MHA with queries, keys & values."""
    #print ('Pre', query.shape, key.shape, value.shape)
    query_heads = self._linear_projection(query, self.key_size, "query")
    key_heads = self._linear_projection(key, self.key_size, "key")
    value_heads = self._linear_projection(value, self.value_size, "value")
    #print ('Post', query_heads.shape, key_heads.shape, value_heads.shape)

    attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
    sqrt_key_size = jnp.sqrt(self.key_size).astype(key.dtype)
    attn_logits = attn_logits / sqrt_key_size

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


class TriangleMultiplication(hk.Module):
  """Triangle multiplication layer ("outgoing" or "incoming").
  Jumper et al. (2021) Suppl. Alg. 11 "TriangleMultiplicationOutgoing"
  Jumper et al. (2021) Suppl. Alg. 12 "TriangleMultiplicationIncoming"
  """

  def __init__(self, channels, outgoing=False, name='triangle_multiplication'):
    super().__init__(name=name)
    self.channels = channels
    self.outgoing = outgoing

  def __call__(self, act, mask, is_training=True):
    """Builds TriangleMultiplication module.
    Arguments:
      act: Pair activations, shape [N_res, N_res, c_z]
      mask: Pair mask, shape [N_res, N_res].
      is_training: Whether the module is in training mode.
    Returns:
      Outputs, same shape/type as act.
    """

    act = hk.LayerNorm(axis=[-1], create_scale=True, create_offset=True,
                       name='layer_norm_input')(act)
    input_act = act

    left_projection = Linear(self.channels, name='left_projection')
    left_proj_act = mask * left_projection(act)

    right_projection = Linear(self.channels, name='right_projection')
    right_proj_act = mask * right_projection(act)

    left_gate_values = jax.nn.sigmoid(
        Linear(self.channels, name='left_gate')(act))
    right_gate_values = jax.nn.sigmoid(
        Linear(self.channels, name='right_gate')(act))

    left_proj_act *= left_gate_values
    right_proj_act *= right_gate_values

    # "Outgoing" edges equation: 'ikc,jkc->ijc'
    # "Incoming" edges equation: 'kjc,kic->ijc'
    # Note on the Suppl. Alg. 11 & 12 notation:
    # For the "outgoing" edges, a = left_proj_act and b = right_proj_act
    # For the "incoming" edges, it's swapped:
    #   b = left_proj_act and a = right_proj_act
    if self.outgoing: 
        equation = 'ikc,jkc->ijc'
        act = jnp.einsum(equation, left_proj_act, right_proj_act)
    else: 
        equation = 'kjc,kic->ijc'
        act = jnp.einsum(equation, right_proj_act, left_proj_act)

    act = hk.LayerNorm(axis=[-1], create_scale=True, 
        create_offset=True, name='center_layer_norm')(act)

    output_channel = int(input_act.shape[-1])

    act = Linear(output_channel, initializer=utils.final_init(gc),
        name='output_projection')(act)

    gate_values = jax.nn.sigmoid(
        Linear(output_channel, name='gating_linear')(input_act))
    act *= gate_values

    return act


class TriangleAttention(hk.Module):
  """Triangle multihead attention."""

  def __init__(self, num_heads, output_dim, name='attention'):
    super().__init__(name=name)

    self.num_heads = num_heads
    self.output_dim = output_dim

  def __call__(self, q_data, m_data, bias):
    """Builds Attention module.
    Arguments:
      q_data: A tensor of queries, shape [batch_size, N_queries, q_channels].
      m_data: A tensor of memories from which the keys and values are
        projected, shape [batch_size, N_keys, m_channels].
      bias: A bias for the attention, shape [batch_size, N_queries, N_keys].
      nonbatched_bias: Shared bias, shape [N_queries, N_keys].
    Returns:
      A float32 tensor of shape [batch_size, N_queries, output_dim].
    """
    # Default attention
    q_weights = hk.get_parameter(
        'query_w', shape=(q_data.shape[-1], self.num_head, self.output_dim),
        init=glorot_uniform())
    k_weights = hk.get_parameter(
        'key_w', shape=(m_data.shape[-1], self.num_head, self.output_dim),
        init=glorot_uniform())
    v_weights = hk.get_parameter(
        'value_w', shape=(m_data.shape[-1], self.num_head, self.output_dim),
        init=glorot_uniform())

    q = jnp.einsum('bqa,ahc->bqhc', q_data, q_weights) * key_dim**(-0.5)
    k = jnp.einsum('bka,ahc->bkhc', m_data, k_weights)
    v = jnp.einsum('bka,ahc->bkhc', m_data, v_weights)
    logits = jnp.einsum('bqhc,bkhc->bhqk', q, k) + bias
    if nonbatched_bias is not None:
      logits += jnp.expand_dims(nonbatched_bias, axis=0)
    weights = jax.nn.softmax(logits)
    weighted_avg = jnp.einsum('bhqk,bkhc->bqhc', weights, v)

    # Gate updating
    gating_weights = hk.get_parameter(
        'gating_w', shape=(q_data.shape[-1], num_head, value_dim),
        init=hk.initializers.Constant(0.0))
    gating_bias = hk.get_parameter(
        'gating_b',
        shape=(num_head, value_dim),
        init=hk.initializers.Constant(1.0))

    gate_values = jnp.einsum('bqc, chv->bqhv', q_data,
                             gating_weights) + gating_bias
    
    gate_values = jax.nn.sigmoid(gate_values)

    # Gate application
    weighted_avg *= gate_values

    init = glorot_uniform()
    o_weights = hk.get_parameter(
        'output_w', shape=(num_head, value_dim, self.output_dim),
        init=init)
    o_bias = hk.get_parameter('output_b', shape=(self.output_dim,),
                              init=hk.initializers.Constant(0.0))

    output = jnp.einsum('bqhc,hco->bqo', weighted_avg, o_weights) + o_bias

    return output


