import os
import sys
import jax
import jraph
from functools import partial
import jax.numpy as jnp
import pdb as debug
from jax import vmap

@jax.jit
def distances_from_coords(coords1, coords2):
    return jnp.sqrt(jnp.sum((coords1-coords2)**2, axis=-1))

#@partial(jax.jit, static_argnums=5)
def get_edges(cmap, local1, local2, cloud1, cloud2, enum):
  
  def _get_residue_edges(line):

    def _next_lowest(min_val, array):
      array = jnp.where(array>min_val, array, array+10e6)
      next_val = jnp.min(array)
      return next_val, jnp.argwhere(array==next_val, size=1)
    
    sorting_input = jnp.broadcast_to(line[None,:], (enum, line.shape[0]))
    sorted_idxs = jnp.squeeze(jax.lax.scan(_next_lowest, 0, sorting_input)[1])
    sorted_line = line[sorted_idxs]
    return sorted_line, sorted_idxs
  
  def _get_relative_features(local_i, local_j, coord_i, coord_j):
    basis = jnp.stack(local_i, axis=0)
    q_ij = jnp.matmul(basis, (coord_j-coord_i))
    k_ij = jnp.matmul(basis, local_j[0])
    t_ij = jnp.matmul(basis, local_j[1])
    s_ij = jnp.matmul(basis, local_j[2])
    feats = jnp.concatenate((q_ij, k_ij, t_ij, s_ij), axis=0)
    return jnp.transpose(feats)

  def _get_distance_features(dist):
    scales = 1.5 ** (jnp.indices((15,))[0]+1)
    return jnp.exp(-((dist ** 2) / scales))

  nodes_i = jnp.indices((cmap.shape[0],))[0]
  nodes_i = jnp.broadcast_to(nodes_i[:,None], (cmap.shape[0], enum))
  dist, neighs_j = vmap(_get_residue_edges)(cmap)
  dist, nodes_i, neighs_j = jnp.ravel(dist), jnp.ravel(nodes_i), jnp.ravel(neighs_j),

  local_i, cloud_i = local1[nodes_i], cloud1[nodes_i]
  local_j, cloud_j = local2[neighs_j], cloud2[neighs_j]
  rfeats = vmap(_get_relative_features)(local_i, local_j, cloud_i, cloud_j)
  dfeats = vmap(_get_distance_features)(dist)
  edges = jnp.concatenate((rfeats, dfeats), axis=-1)

  return (edges, nodes_i, neighs_j)

### TODO move in environment?
def initialize_clouds(cloud1, cloud2, cmap1, cmap2, key):
    
    def cloud_center(cloud):
        return jnp.sum(cloud, axis=0)/cloud.shape[0]

    def init_cloud(cloud, cm, key):
        cloud = cloud-cm[None,:]
        quat = quat_from_pred(jax.random.normal(key, (3,)))
        cloud = quat_rotation(cloud, quat)
        tgroup = [cm, quat]
        return cloud, tgroup

    cm1 = cloud_center(cloud1)
    cm2 = cloud_center(cloud2)

    max_c1 = jnp.max(cmap1)
    max_c2 = jnp.max(cmap2)
    cm2 -= jnp.array([max_c1+max_c2, 0., 0.])

    key1, key2 = jax.random.split(key, 2)
    cloud1, tgroup1 = init_cloud(cloud1, cm1, key1)
    cloud2, tgroup2 = init_cloud(cloud2, cm2, key2)
    return cloud1, cloud2, tgroup1, tgroup2

### TODO WRITE AS A CLASS
def write_mmcif(chain_ids, transforms, in_path, out_path) -> None:
    io = MMCIFIO()
    pdbp = PDBParser()
    cifp = MMCIFParser()
    struc1 = pdbp.get_structure('', in_path[0])
    struc2 = pdbp.get_structure('', in_path[1])
    
    if os.path.exists(out_path):
        out_struc = cifp.get_structure('-', out_path)
        for model in out_struc: 
            last_model_id = model.get_id() + 1
        new_model_id = last_model_id+1
    else:
        out_struc = Structure('-')
        new_model_id = 1

    out_model = Model(new_model_id)
    out_struc.add(out_model)
    out_model.set_parent(out_struc)

    for rot, tr in transforms:
        mid, cid = ids.pop(0)
        if mid in struc1:
            if cid in struc1[mid]: in_chain = struc1[mid][cid]
        if mid in struc2:
            if cid in struc2[mid]: in_chain = struc2[mid][cid]
        cloud = [atom.get_coord() for atom in unfold_entities(in_chain, 'A')]
        cloud = jnp.array(cloud)
        cloud = jnp.matmul(rot, cloud.transpose()).transpose() - tr
        for new, atom in zip(cloud, unfold_entities(in_chain, 'A')): 
            atom.set_coord(new)

        in_chain.set_parent(out_model)
        out_model.add(in_chain)

    for chain in unfold_entities(out_struc, 'C'):
        print (chain.get_full_id())

    io.set_structure(out_struc)
    io.save(out_path)
