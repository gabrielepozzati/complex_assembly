import os
import sys
import jax
import jraph
import functools
import jax.numpy as jnp
import pdb as debug

@functools.partial(jax.jit, static_argnums=1)
def get_edges(cmap, enum):
  
  def _residue_edges(line):

    def _next_lowest(min_val, array):
      array = jnp.where(array>min_val, array, array+10e6)
      next_val = jnp.min(array)
      return next_val, jnp.argwhere(array==next_val, size=1)
    
    sorting_input = jnp.broadcast_to(line[None,:], (enum, line.shape[0]))
    sorted_idxs = jnp.squeeze(jax.lax.scan(_next_lowest, 0, sorting_input)[1])
    sorted_line = line[sorted_idxs]
    return sorted_line, sorted_idxs
  
  senders = jnp.indices((cmap.shape[0],))[0]
  senders = jnp.broadcast_to(senders[:,None], (cmap.shape[0], enum))
  edges, receivers = jax.vmap(_residue_edges, in_axes=0, out_axes=0)(cmap)
  return (jnp.ravel(edges), 
          jnp.ravel(senders), 
          jnp.ravel(receivers))

def get_interface_edges(icmap, enum, pad):

  length1 = icmap.shape[0]
  length2 = icmap.shape[1]
  padlen1 = (pad-length1)*enum
  padlen2 = (pad-length2)*enum

  edges12, senders12, receivers12 = get_edges(icmap, enum)
  edges21, senders21, receivers21 = get_edges(jnp.transpose(icmap), enum)
  senders21, receivers12 = senders21+pad, receivers12+pad

  edges12 = jnp.pad(edges12, (0, padlen1))
  edges21 = jnp.pad(edges21, (0, padlen2))

  senders12 = jnp.pad(senders12, (0, padlen1), constant_values=length1)
  senders21 = jnp.pad(senders21, (0, padlen2), constant_values=length2+pad)

  receivers12 = jnp.pad(receivers12, (0, padlen1), constant_values=length2+pad)
  receivers21 = jnp.pad(receivers21, (0, padlen2), constant_values=length1)

  return edges12, senders12, receivers12, edges21, senders21, receivers21
