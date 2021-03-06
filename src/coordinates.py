import os
import sys
import jax.numpy as jnp
from typing import Dict, Union, Optional

from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Selection import unfold_entities
from scipy.spatial.transform import Rotation as R

def initialize_clouds(clouds: list, seed: int) -> Dict[str, jnp.ndarray]:

    tgroup = []
    rotated_chains = []
    for idx, chain in enumerate(clouds):
        rotation = R.random(random_state=seed).as_matrix()
        chain = jnp.matmul(rotation, chain.transpose()).transpose()
        cm = jnp.sum(chain, axis=0)/chain.shape[0]
        rotated_chains.append(chain - cm)
        tgroup.append([rotation, cm])
    return rotated_chains, tgroup

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
