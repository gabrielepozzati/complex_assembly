import os
import sys
import jax
import jax.numpy as jnp
from typing import Dict, Union, Optional
from ops import *

from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Selection import unfold_entities


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
