
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


class DenseBlock(hk.Module):
  """A 2-layer MLP"""
  def __init__(self,
      #init_scale: float,
      widening_factor: int = 4,
      name: Optional[str] = None):
    super().__init__(name=name)
    self._init_scale = init_scale
    self._widening_factor = widening_factor

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    channels = x.shape[-1]
    #initializer = hk.initializers.VarianceScaling(self._init_scale)
    x = Linear(self._widening_factor * channels)(x)
    x = jax.nn.gelu(x)
      
    return Linear(channels)(x)


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
    sqrt_key_size = np.sqrt(self.key_size).astype(key.dtype)
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

class Transformer(hk.Module):
    """A transformer stack."""
    def __init__(self,
                 num_heads: int,
                 num_layers: int,
                 dropout_rate: float,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._dropout_rate = dropout_rate

    def __call__(self,
                 h: jnp.ndarray,
                 mask: Optional[jnp.ndarray],
                 is_training: bool) -> jnp.ndarray:
        """Connects the transformer.
        Args:
          h: Inputs, [B, T, H].
          mask: Padding mask, [B, T].
          is_training: Whether we're training or not.
        Returns:
          Array of shape [B, T, H].
        """
        init_scale = 2. / self._num_layers
        dropout_rate = self._dropout_rate if is_training else 0.
        for i in range(self._num_layers):
            h_norm = layer_norm(h, name=f'h{i}_ln_1')
            h_attn = SelfAttention(
                num_heads=self._num_heads,
                key_size=64,
                w_init_scale=init_scale,
                name=f'h{i}_attn')(h_norm, mask=mask)
            h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
            h = h + h_attn
            h_norm = layer_norm(h, name=f'h{i}_ln_2')
            h_dense = DenseBlock(init_scale, name=f'h{i}_mlp')(h_norm)
            h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
            h = h + h_dense
        h = layer_norm(h, name='ln_f')
        return h

def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    """Apply a unique LayerNorm to x with default settings."""
    return hk.LayerNorm(axis=-1,
                        create_scale=True,
                        create_offset=True,
                        name=name)(x)

