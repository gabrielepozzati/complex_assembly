import os
import sys
import jax.numpy as jnp
from typing import Dict, Union, Optional

from Bio.PDB.Model import Model
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Selection import unfold_entities
from scipy.spatial.transform import Rotation as R

def initialize_clouds(data: Dict[str, jnp.ndarray], 
                      seed: int) -> Dict[str, jnp.ndarray]:

    for idx, chain in enumerate(data):
        rotation = R.random(random_state=seed).as_matrix()
        data[chain] = jnp.matmul(rotation, data[chain].transpose()).transpose()
        cm = jnp.sum(data[chain], axis=0)/data[chain].shape[0]
        data[chain] = data[chain] - cm
    return data

def write_mmcif(data: dict, out_path: str) -> None:
    io = MMCIFIO()
    pdbp = PDBParser()
    cifp = MMCIFParser()
    struc = pdbp.get_structure('', data['path'])
    
    if os.path.exists(out_path):
        out_struc = cifp.get_structure('', out_path)
        for model in out_struc: last_model_id = model.get_id()
        new_model_id = last_model_id + 1
    else:
        out_struc = Structure('x')
        new_model_id = 1

    out_model = Model(new_model_id)
    out_struc.add(out_model)
    out_model.set_parent(out_struc)
    for chain in data['chains']:
        cloud = data['chains'][chain]
        id_mask = data['id_masks'][chain]
        for coord, (sid, mid, cid, rid, aid) in zip(cloud, id_mask):
            struc[mid][cid][rid][aid].set_coord(coord)
        in_chain = struc[mid][cid]
        in_chain.set_parent(out_model)
        out_model.add(in_chain)
    
    io.set_structure(struc)
    io.save()

    



    
