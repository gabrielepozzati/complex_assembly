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


def initialize_clouds(cloud, key):

    cloud = jnp.array(cloud)
    cm = jnp.sum(cloud, axis=0)/cloud.shape[0]
    cloud = cloud - cm

    quat = quat_from_pred(jax.random.normal(key, (3,)))
    cloud = quat_rotation(cloud, quat)
    tgroup = [cm, quat]

    return cloud, tgroup

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
