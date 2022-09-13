import functools
from typing import Any, Callable, Iterable, Mapping, Optional, Union

import jax
import jax.numpy as jnp
import jax.tree_util as tree
from jraph._src import graph as gn_graph
from jraph._src import utils

# As of 04/2020 pytype doesn't support recursive types.
# pytype: disable=not-supported-yet
ArrayTree = Union[jnp.ndarray, Iterable['ArrayTree'], Mapping[Any, 'ArrayTree']]

# All features will be an ArrayTree.
NodeFeatures = EdgeFeatures = SenderFeatures = ReceiverFeatures = Globals = ArrayTree

# Signature:
# (edges of each node to be aggregated, segment ids, number of segments) ->
# aggregated edges
AggregateEdgesToNodesFn = Callable[
    [EdgeFeatures, jnp.ndarray, int], NodeFeatures]

# Signature:
# (nodes of each graph to be aggregated, segment ids, number of segments) ->
# aggregated nodes
AggregateNodesToGlobalsFn = Callable[[NodeFeatures, jnp.ndarray, int],
                                     Globals]

# Signature:
# (edges of each graph to be aggregated, segment ids, number of segments) ->
# aggregated edges
AggregateEdgesToGlobalsFn = Callable[[EdgeFeatures, jnp.ndarray, int],
                                     Globals]

# Signature:
# (edge features, sender node features, receiver node features, globals) ->
# attention weights
AttentionLogitFn = Callable[
    [EdgeFeatures, SenderFeatures, ReceiverFeatures, Globals], ArrayTree]

# Signature:
# (edge features, weights) -> edge features for node update
AttentionReduceFn = Callable[[EdgeFeatures, ArrayTree], EdgeFeatures]

# Signature:
# (edges to be normalized, segment ids, number of segments) ->
# normalized edges
AttentionNormalizeFn = Callable[[EdgeFeatures, jnp.ndarray, int], EdgeFeatures]

# Signature:
# (edge features, sender node features, receiver node features, globals) ->
# updated edge features
GNUpdateEdgeFn = Callable[
    [EdgeFeatures, SenderFeatures, ReceiverFeatures, Globals], EdgeFeatures]

# Signature:
# (node features, outgoing edge features, incoming edge features,
#  globals) -> updated node features
GNUpdateNodeFn = Callable[
    [NodeFeatures, SenderFeatures, ReceiverFeatures, Globals], NodeFeatures]

GNUpdateGlobalFn = Callable[[NodeFeatures, EdgeFeatures, Globals], Globals]

def GraphNetwork(
    update_edge_fn = None,
    update_node_fn = True,
    update_global_fn = None,
    aggregate_edges_for_nodes_fn = utils.segment_sum,
    aggregate_nodes_for_globals_fn = utils.segment_sum,
    aggregate_edges_for_globals_fn = utils.segment_sum):
  
  def _ApplyGraphNet(graph):
    """Applies a configured GraphNetwork to a graph.
    This implementation follows Algorithm 1 in https://arxiv.org/abs/1806.01261
    There is one difference. For the nodes update the class aggregates over the
    sender edges and receiver edges separately. This is a bit more general
    the algorithm described in the paper. The original behaviour can be
    recovered by using only the receiver edge aggregations for the update.
    In addition this implementation supports softmax attention over incoming
    edge features.
    Many popular Graph Neural Networks can be implemented as special cases of
    GraphNets, for more information please see the paper.
    Args:
      graph: a `GraphsTuple` containing the graph.
    Returns:
      Updated `GraphsTuple`.
    """
    # pylint: disable=g-long-lambda
    nodes, edges, receivers, senders, globals_, n_node, n_edge = graph
    # Equivalent to jnp.sum(n_node), but jittable
    sum_n_node = tree.tree_leaves(nodes)[0].shape[0]
    print ('sum n node', sum_n_node)
    sum_n_edge = senders.shape[0]
    print ('sum n node', sum_n_edge)

    sent_attributes = tree.tree_map(lambda n: n[senders], nodes)
    print ('sent_attributes', sent_attributes)
    received_attributes = tree.tree_map(lambda n: n[receivers], nodes)
    print ('received_attributes', received_attributes)
    # Here we scatter the global features to the corresponding edges,
    # giving us tensors of shape [num_edges, global_feat].
    global_edge_attributes = tree.tree_map(lambda g: jnp.repeat(
        g, n_edge, axis=0, total_repeat_length=sum_n_edge), globals_)
    print ('global_edge_attributes', global_edge_attributes)

    if update_node_fn:
      sent_attributes = tree.tree_map(
          lambda e: aggregate_edges_for_nodes_fn(e, senders, sum_n_node), edges)
      print ('sent_attributes', sent_attributes.shape, sent_attributes)
      received_attributes = tree.tree_map(
          lambda e: aggregate_edges_for_nodes_fn(e, receivers, sum_n_node),
          edges)
      # Here we scatter the global features to the corresponding nodes,
      # giving us tensors of shape [num_nodes, global_feat].
      global_attributes = tree.tree_map(lambda g: jnp.repeat(
          g, n_node, axis=0, total_repeat_length=sum_n_node), globals_)
      print ('global_attributes', global_attributes.shape, global_attributes)
      nodes = update_node_fn(nodes, sent_attributes,
                             received_attributes, global_attributes)

    
    if update_global_fn:
      n_graph = n_node.shape[0]
      graph_idx = jnp.arange(n_graph)
      # To aggregate nodes and edges from each graph to global features,
      # we first construct tensors that map the node to the corresponding graph.
      # For example, if you have `n_node=[1,2]`, we construct the tensor
      # [0, 1, 1]. We then do the same for edges.
      node_gr_idx = jnp.repeat(
          graph_idx, n_node, axis=0, total_repeat_length=sum_n_node)
      edge_gr_idx = jnp.repeat(
          graph_idx, n_edge, axis=0, total_repeat_length=sum_n_edge)
      # We use the aggregation function to pool the nodes/edges per graph.
      node_attributes = tree.tree_map(
          lambda n: aggregate_nodes_for_globals_fn(n, node_gr_idx, n_graph),
          nodes)
      edge_attribtutes = tree.tree_map(
          lambda e: aggregate_edges_for_globals_fn(e, edge_gr_idx, n_graph),
          edges)
      # These pooled nodes are the inputs to the global update fn.
      globals_ = update_global_fn(node_attributes, edge_attribtutes, globals_)
    # pylint: enable=g-long-lambda
    return gn_graph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        receivers=receivers,
        senders=senders,
        globals=globals_,
        n_node=n_node,
        n_edge=n_edge)

  return _ApplyGraphNet

