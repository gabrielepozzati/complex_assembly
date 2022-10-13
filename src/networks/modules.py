
import numbers
import functools
import haiku as hk
import numpy as np

import jax
import jax.numpy as jnp
import jax.tree_util as tree
from jax import grad, jit, vmap
from jax import random

import jraph
from jraph._src.models import *
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


class MultiLayerPerceptron(hk.Module):

  def __init__(self, num_channels, out_channels, num_layers, name='MLP'):

    super().__init__(name=name)
    self.stack = [Linear(num_channels) for _ in range(num_layers)]
    self.out_layer = Linear(out_channels)

  def __call__(self, x):

    for layer in self.stack:
      x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
      x = jax.nn.relu(layer(x))
      
    x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
    return jax.nn.relu(self.out_layer(x))


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

  def __init__(self, channels, outgoing=True, name='triangle_multiplication'):
    super().__init__(name=name)
    self.channels = channels
    self.outgoing = outgoing

  def __call__(self, pair_data):
    """Builds TriangleMultiplication module.
    Arguments:
      act: Pair activations, shape [N_res, N_res, c_z]
      mask: Pair mask, shape [N_res, N_res].
      is_training: Whether the module is in training mode.
    Returns:
      Outputs, same shape/type as act.
    """
   
    pair_data = hk.LayerNorm(axis=[-1], create_scale=True, 
        create_offset=True, name='layer_norm_input')(pair_data)
    
    # shortcut
    input_data = pair_data

    # extracting first two indexes
    left_projection = Linear(self.channels, name='left_projection')
    left_proj_act = left_projection(pair_data)

    right_projection = Linear(self.channels, name='right_projection')
    right_proj_act = right_projection(jnp.transpose(pair_data, -2, -3))

    # gating first two indexes
    left_gate_values = jax.nn.sigmoid(
        Linear(self.channels, name='left_gate')(pair_data))
    right_gate_values = jax.nn.sigmoid(
        Linear(self.channels, name='right_gate')(jnp.transpose(pair_data, -2, -3)))

    left_proj_act *= left_gate_values
    right_proj_act *= right_gate_values

    # triangulation
    if self.outgoing:
        if self.squared: equation = 'ikc,jkc->ijc'
        else: equation = 'ikc,kjc->ijc'
        pair_data = jnp.einsum(equation, left_proj_act, right_proj_act)
    else:
        if self.squared: equation = 'kjc,kic->ijc'
        else: equation = 'kjc,ikc->ijc'
        pair_data = jnp.einsum(equation, right_proj_act, left_proj_act)

    pair_data = hk.LayerNorm(axis=[-1], create_scale=True, 
        create_offset=True, name='center_layer_norm')(pair_data)

    # gating with shortcut
    output_channel = int(input_act.shape[-1])
    pair_data = Linear(output_channel, name='output_projection')(pair_data)

    gate_values = jax.nn.sigmoid(
        Linear(output_channel, name='gating_linear')(input_data))
    
    pair_data *= gate_values

    return pair_data


class TriangleAttention(hk.Module):
  """Triangle multihead attention."""

  def __init__(self, num_heads, channels, 
               end_node=False, squared=False, name='attention'):

    super().__init__(name=name)
    self.num_heads = num_heads
    self.channels = channels
    self.end_node = end_node

  def __call__(self, pair_data):
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
    # swap to end node triangle attention
    if not end_node: pair_data = jnp.swapaxes(pair_data, -2, -3)

    # query, keys and values normalization
    pair_data = hk.LayerNorm(axis=[-1], create_scale=True, 
        create_offset=True, name='qkv_norm')(pair_data)

    # computing pair representation bias
    init_factor = 1. / jnp.sqrt(int(pair_data.shape[-1]))
    weights = hk.get_parameter(
        'feat_2d_weights', shape=(pair_data.shape[-1], self.num_head),
        init=hk.initializers.RandomNormal(stddev=init_factor))
    uniq_bias = jnp.einsum('qkc,ch->hqk', pair_data, weights)

    # default attention
    if squared: pair_trans = pair_data
    else: pair_trans = jnp.swapaxes(pair_data, -2, -3)

    q_weights = hk.get_parameter(
        'query_w', shape=(pair_data.shape[-1], self.num_head, self.channels),
        init=glorot_uniform())
    k_weights = hk.get_parameter(
        'key_w', shape=(pair_trans.shape[-1], self.num_head, self.channels),
        init=glorot_uniform())
    v_weights = hk.get_parameter(
        'value_w', shape=(pair_trans.shape[-1], self.num_head, self.channels),
        init=glorot_uniform())

    q = jnp.einsum('bqa,ahc->bqhc', pair_data, q_weights) * self.channels**(-0.5)
    k = jnp.einsum('bka,ahc->bkhc', pair_trans, k_weights)
    v = jnp.einsum('bka,ahc->bkhc', pair_trans, v_weights)
    
    logits = jnp.einsum('bqhc,bkhc->bhqk', q, k) + uniq_bias
    weights = jax.nn.softmax(logits)
    weighted_avg = jnp.einsum('bhqk,bkhc->bqhc', weights, v)

    # gate updating
    gating_weights = hk.get_parameter(
        'gating_w', shape=(q_data.shape[-1], num_head, value_dim),
        init=hk.initializers.Constant(0.0))

    gating_bias = hk.get_parameter(
        'gating_b', shape=(num_head, value_dim),
        init=hk.initializers.Constant(1.0))

    gate_values = jnp.einsum('bqc, chv->bqhv', q_data,
                             gating_weights) + gating_bias
    
    gate_values = jax.nn.sigmoid(gate_values)

    # gate application
    weighted_avg *= gate_values

    init = glorot_uniform()
    
    o_weights = hk.get_parameter(
        'output_w', shape=(num_head, value_dim, self.channels),
        init=init)
    
    o_bias = hk.get_parameter('output_b', shape=(self.channels,),
                              init=hk.initializers.Constant(0.0))

    output = jnp.einsum('bqhc,hco->bqo', weighted_avg, o_weights) + o_bias

    # swap axis back to original
    if not end_node: pair_data = jnp.swapaxes(pair_data, -2, -3)

    return output


class TriangleModule(hk.Module):
  def __init__(self, num_heads, num_channels, squared=False):
    edge_ta_start = TriangleAttention(
        num_heads, num_channels, squared=squared)

    edge_ta_end = TriangleAttention(
        num_heads, num_channels, end_node=True, squared=squared)

    edge_tm_out = TriangleMultiplication(num_channels)
    
    edge_tm_in = TriangleMultiplication(num_channels, outgoing=False)

  def __call__(self, pair_data):

    pair_data = edge_ta_start(pair_data)
    pair_data = edge_ta_end(pair_data)
    pair_data = edge_tm_out(pair_data)
    pair_data = edge_tm_in(pair_data)
     
    return pair_data


class InvariantPointAttention(hk.Module):
  """Invariant Point attention module.
  The high-level idea is that this attention module works over a set of points
  and associated orientations in 3D space (e.g. protein residues).
  Each residue outputs a set of queries and keys as points in their local
  reference frame.  The attention is then defined as the euclidean distance
  between the queries and keys in the global frame.
  Jumper et al. (2021) Suppl. Alg. 22 "InvariantPointAttention"
  """

  def __init__(self, num_heads, num_channels,
               num_scalar_qk, num_scalar_v,
               num_point_qk, num_point_v,
               dist_epsilon=1e-8, name='invariant_point_attention'):
    """Initialize.
    Args:
      config: Structure Module Config
      global_config: Global Config of Model.
      dist_epsilon: Small value to avoid NaN in distance calculation.
      name: Haiku Module name.
    """
    super().__init__(name=name)
    self.num_heads = num_heads
    self.num_scalar_qk = num_scalar_qk
    self.num_point_qk = num_point_qk
    self.num_scalar_v = num_scalar_v
    self.num_point_v = num_point_v
    self.num_channel = num_channels
    self._dist_epsilon = dist_epsilon
    self._zero_initialize_last = False

  def __call__(self, inputs_1d, inputs_2d, cloud):
    """Compute geometry-aware attention.
    Given a set of query residues (defined by affines and associated scalar
    features), this function computes geometry-aware attention between the
    query residues and target residues.
    The residues produce points in their local reference frame, which
    are converted into the global frame in order to compute attention via
    euclidean distance.
    Equivalently, the target residues produce points in their local frame to be
    used as attention values, which are converted into the query residues'
    local frames.
    Args:
      inputs_1d: (N, C) 1D input embedding that is the basis for the
        scalar queries.
      inputs_2d: (N, M, C') 2D input embedding, used for biases and values.
      mask: (N, 1) mask to indicate which elements of inputs_1d participate
        in the attention.
      affine: QuatAffine object describing the position and orientation of
        every element in inputs_1d.
    Returns:
      Transformation of the input embedding.
    """
    num_residues, _ = inputs_1d.shape

    # Improve readability by removing a large number of 'self's.
    num_head = self.num_heads
    num_scalar_qk = self.num_scalar_qk
    num_point_qk = self.num_point_qk
    num_scalar_v = self.num_scalar_v
    num_point_v = self.num_point_v
    num_output = self.num_channel

    # Construct scalar queries of shape:
    # [num_query_residues, num_head, num_points]
    q_scalar = Linear(
        num_head * num_scalar_qk, name='q_scalar')(
            inputs_1d)
    q_scalar = jnp.reshape(
        q_scalar, [num_residues, num_head, num_scalar_qk])

    # Construct scalar keys/values of shape:
    # [num_target_residues, num_head, num_points]
    kv_scalar = Linear(
        num_head * (num_scalar_v + num_scalar_qk), name='kv_scalar')(
            inputs_1d)
    kv_scalar = jnp.reshape(kv_scalar,
                            [num_residues, num_head,
                             num_scalar_v + num_scalar_qk])
    k_scalar, v_scalar = jnp.split(kv_scalar, [num_scalar_qk], axis=-1)

    # Construct query points of shape:
    # [num_residues, num_head, num_point_qk]

    # First construct query points in local frame.
    q_point_local = Linear(
        num_head * 3 * num_point_qk, name='q_point_local')(
            inputs_1d)
    q_point_local = jnp.split(q_point_local, 3, axis=-1)
    # Project query points into global frame.
    q_point_global = traslate_point(q_point_local, cloud, extra_dims=0)
    # Reshape query point for later use.
    q_point = [
        jnp.reshape(x, [num_residues, num_head, num_point_qk])
        for x in q_point_global]

    # Construct key and value points.
    # Key points have shape [num_residues, num_head, num_point_qk]
    # Value points have shape [num_residues, num_head, num_point_v]

    # Construct key and value points in local frame.
    kv_point_local = Linear(
        num_head * 3 * (num_point_qk + num_point_v), name='kv_point_local')(
            inputs_1d)
    kv_point_local = jnp.split(kv_point_local, 3, axis=-1)
    # Project key and value points into global frame.
    kv_point_global = traslate_point(kv_point_local, cloud, extra_dims=1)
    kv_point_global = [
        jnp.reshape(x, [num_residues,
                        num_head, (num_point_qk + num_point_v)])
        for x in kv_point_global]
    # Split key and value points.
    k_point, v_point = list(
        zip(*[
            jnp.split(x, [num_point_qk,], axis=-1)
            for x in kv_point_global
        ]))

    # We assume that all queries and keys come iid from N(0, 1) distribution
    # and compute the variances of the attention logits.
    # Each scalar pair (q, k) contributes Var q*k = 1
    scalar_variance = max(num_scalar_qk, 1) * 1.
    # Each point pair (q, k) contributes Var [0.5 ||q||^2 - <q, k>] = 9 / 2
    point_variance = max(num_point_qk, 1) * 9. / 2

    # Allocate equal variance to scalar, point and attention 2d parts so that
    # the sum is 1.

    num_logit_terms = 3

    scalar_weights = np.sqrt(1.0 / (num_logit_terms * scalar_variance))
    point_weights = np.sqrt(1.0 / (num_logit_terms * point_variance))
    attention_2d_weights = np.sqrt(1.0 / (num_logit_terms))

    # Trainable per-head weights for points.
    trainable_point_weights = jax.nn.softplus(hk.get_parameter(
        'trainable_point_weights', shape=[num_head],
        # softplus^{-1} (1)
        init=hk.initializers.Constant(np.log(np.exp(1.) - 1.))))
    point_weights *= jnp.expand_dims(trainable_point_weights, axis=1)

    v_point = [jnp.swapaxes(x, -2, -3) for x in v_point]
    q_point = [jnp.swapaxes(x, -2, -3) for x in q_point]
    k_point = [jnp.swapaxes(x, -2, -3) for x in k_point]

    dist2 = [
        squared_difference(qx[:, :, None, :], kx[:, None, :, :])
        for qx, kx in zip(q_point, k_point)
    ]
    dist2 = sum(dist2)

    attn_qk_point = -0.5 * jnp.sum(
        point_weights[:, None, None, :] * dist2, axis=-1)

    v = jnp.swapaxes(v_scalar, -2, -3)
    q = jnp.swapaxes(scalar_weights * q_scalar, -2, -3)
    k = jnp.swapaxes(k_scalar, -2, -3)
    attn_qk_scalar = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_qk_scalar + attn_qk_point

    attention_2d = Linear(num_head, name='attention_2d')(inputs_2d)

    attention_2d = jnp.transpose(attention_2d, [2, 0, 1])
    attention_2d = attention_2d_weights * attention_2d
    attn_logits += attention_2d

    #mask_2d = mask * jnp.swapaxes(mask, -1, -2)
    #attn_logits -= 1e5 * (1. - mask_2d)

    # [num_head, num_query_residues, num_target_residues]
    attn = jax.nn.softmax(attn_logits)

    # [num_head, num_query_residues, num_head * num_scalar_v]
    result_scalar = jnp.matmul(attn, v)

    # For point result, implement matmul manually so that it will be a float32
    # on TPU.  This is equivalent to
    # result_point_global = [jnp.einsum('bhqk,bhkc->bhqc', attn, vx)
    #                        for vx in v_point]
    # but on the TPU, doing the multiply and reduce_sum ensures the
    # computation happens in float32 instead of bfloat16.
    result_point_global = [jnp.sum(
        attn[:, :, :, None] * vx[:, None, :, :],
        axis=-2) for vx in v_point]

    # [num_query_residues, num_head, num_head * num_(scalar|point)_v]
    result_scalar = jnp.swapaxes(result_scalar, -2, -3)
    result_point_global = [
        jnp.swapaxes(x, -2, -3)
        for x in result_point_global]

    # Features used in the linear output projection. Should have the size
    # [num_query_residues, ?]
    output_features = []

    result_scalar = jnp.reshape(
        result_scalar, [num_residues, num_head * num_scalar_v])
    output_features.append(result_scalar)

    result_point_global = [
        jnp.reshape(r, [num_residues, num_head * num_point_v])
        for r in result_point_global]
    result_point_local = traslate_point(result_point_global, -(cloud), extra_dims=1)
    output_features.extend(result_point_local)

    output_features.append(jnp.sqrt(self._dist_epsilon +
                                    jnp.square(result_point_local[0]) +
                                    jnp.square(result_point_local[1]) +
                                    jnp.square(result_point_local[2])))

    # Dimensions: h = heads, i and j = residues,
    # c = inputs_2d channels
    # Contraction happens over the second residue dimension, similarly to how
    # the usual attention is performed.
    result_attention_over_2d = jnp.einsum('hij, ijc->ihc', attn, inputs_2d)
    num_out = num_head * result_attention_over_2d.shape[-1]
    output_features.append(
        jnp.reshape(result_attention_over_2d,
                    [num_residues, num_out]))

    final_init = 'zeros' if self._zero_initialize_last else 'linear'

    final_act = jnp.concatenate(output_features, axis=-1)

    return Linear(
        num_output,
        initializer=final_init,
        name='output_projection')(final_act)

class GraphEncoding(hk.Module):
    def __init__(self, num_channels, name='GE'):
        super().__init__(name=name)
        n_enc1 = Linear(2**7, num_input_dims=2)
        n_enc2 = MultiLayerPerceptron(num_channels, num_channels, 1)
        e_enc = MultiLayerPerceptron(2**5, num_channels, 2)

    def __call__(g):
        n_rec = n_enc1(g.nodes)
        n_rec = n_enc2(n_rec)
        e_rec = e_enc(g.edges)
        
        return g.replace(nodes=n_rec, edges=e_rec)


class GraphUpdates(hk.Module):
    def __init__(self, num_channels, name='GU'):
        super().__init__(name=name)
        self.edge_updates = MultiLayerPerceptron(num_channels, num_channels, 1)
        self.node_updates = MultiLayerPerceptron(num_channels, num_channels, 1)
        self.global_updates = MultiLayerPerceptron(num_channels, num_channels, 1)
        self.graph_update = GraphNetwork(
            self.up_edges(), 
            self.up_nodes(),
            update_global_fn=self.up_global())

    def __call__(self, g):
        return self.graph_update(g)

    def up_edges(self, edges, senders, receivers, globals_):
        #print (edges.shape, senders.shape, receivers.shape)
        edges = jnp.concatenate([senders, edges, receivers], axis=-1)
        return self.edge_updates(edges)

    def up_nodes(self, nodes, sent_edges, received_edges, globals_):
        #print (nodes.shape, sent_edges.shape, received_edges.shape)
        nodes = jnp.concatenate([nodes, sent_edges, received_edges], axis=-1)
        return self.node_updates(nodes)

    def up_global(self, nodes, edges, globals_):
        globals_ = jnp.concatenate([nodes, edges], axis=-1)
        return self.global_updates(globals_)


class AttentionGraphStack(hk.Module):
    def __init__(self, num_channels, num_heads, node_a=True, edge_a=True, name='GA'):
        super().__init__(name=name)
        self.node_a = node_a
        self.edge_a = edge_a
        if node_a: self.node_a = MultiHeadAttention(num_heads, num_channels)
        if edge_a: self.edge_a = MultiHeadAttention(num_heads, num_channels)
        self.graph_up = GraphUpdates(num_channels)

    def __call__(g1, g2=None):
        if not g2: g2 = g1
        if node_a: 
            nodes = g1.nodes+self.node_a(g1.nodes, g2.nodes, g2.nodes)
            nodes = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(nodes)
            g1 = g1.replace(nodes=nodes)
        if edge_a:
            edges = g1.edges+self.edge_a(g1.edges, g2.edges, g2.edges)
            edges = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(edges)
            g1 = g1.replace(edges=edges)
        return self.graph_up(g1)

def get_closest_edges(cloud, mask, e_enc, n=10):
    surf_mask = jnp.ravel(jnp.where(mask>0.2, 1, 10e6))

    all_edges, all_senders, all_receivers = [], [], []
    for idx, point in enumerate(cloud):
      
      cmap = jnp.sqrt(jnp.sum(
        (cloud[:, :]-point[None, :])**2, axis=-1))
      center = jnp.array([idx]*n)
      closer = jnp.argsort(cmap*surf_mask, axis=0)[:n]
      edges = one_hot_cmap(cmap[closer])
      edges = jax.nn.relu(self.e_enc(edges))
      all_edges += [edges]
      all_senders += [closer]
      all_receivers += [center]

    edges = jnp.concatenate(all_edges, axis=0)
    senders = jnp.concatenate(all_senders, axis=0)
    receivers = jnp.concatenate(all_receivers, axis=0)
    
    return edges, senders, receivers


class RelativeSurfaceEncoding():
  def __init__(self, num_channels, num_heads, name='RelativeSurfaceEncoding'):

    super().__init__(name=name)
    # ligand receptor comparison
    self.inter_chain_comp = MultiHeadAttention(num_heads, num_channels)

  def __call__(self, nodes1, nodes2):
    # compare ligand nodes with receptor nodes
    up_nodes1 = []
    for node in nodes1:
      node = jnp.expand_dims(node, axis=0)
      up_nodes1.append(self.inter_chain_comp(node, nodes2, nodes2)
    return jnp.concatenate(up_nodes1, axis=0)

