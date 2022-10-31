import os
import sys
import jax
import jraph
import functools
import jax.numpy as jnp
import pdb as debug

@jax.jit
def surface_edges(edges, idx_map):
  
  def _residue_edges(line):

    def _next_lowest(min_val, array):
      array = jnp.where(array>min_val, array, array+10e6)
      next_val = jnp.min(array)
      return next_val, jnp.argwhere(array==next_val, size=1)
    
    sorting_input = jnp.broadcast_to(line[None,:], (4, line.shape[0]))
    sorted_idxs = jnp.squeeze(jax.lax.scan(_next_lowest, 0, sorting_input)[1])
    sorted_line = line[sorted_idxs]
    return sorted_line, sorted_idxs
  
  senders = jnp.indices((edges.shape[0],))[0]
  senders = jnp.broadcast_to(senders[:,None], (edges.shape[0], 4))
  edges, receivers = jax.vmap(_residue_edges, in_axes=0, out_axes=0)(edges)
  return jnp.ravel(edges), jnp.ravel(idx_map[senders]), jnp.ravel(idx_map[receivers])

def distance_subgraph(graph, thr=8):
  
  for idx, n in enumerate(range(33,0,-1)):
    if thr == n:
      good_idx = idx
      break

  max_idx = jnp.argmax(graph.edges, axis=-1)
  receptive_field = jnp.argwhere(max_idx>=good_idx)

  return jraph.GraphsTuple(
      nodes=graph.nodes, 
      edges=graph.edges[receptive_field],
      senders=graph.senders[receptive_field],
      receivers=graph.receivers[receptive_field],
      n_node=graph.n_node, n_edge=len(graph.edges[receptive_field]))


def surface_subgraph(graph, surf_mask):
  
  surf_idx = jnp.argwhere(surf_mask>=0.2)
  surf_nodes = graph.nodes[surf_idx]

  edges, senders, receivers = [], [], []
  for s, r, e in zip(graph.senders, graph.receivers, graph.edges):
      if s in surf_idx and r in surf_idx:
          senders.append(s)
          receivers.append(r)
          edges.append(e)
  
  return jraph.GraphsTuple(
      nodes=jnp.array(surf_nodes), 
      edges=jnp.array(edges), 
      senders=jnp.array(senders), 
      receivers=jnp.array(receivers),
      n_node=len(surf_nodes), n_edge=len(edges),
      globals=graph.globals)
