import os
import sys
import glob
import pickle
import jax.numpy as jnp

from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Polypeptide import three_to_one
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Selection import unfold_entities
from coordinates import *

standard_residues = [
    'A','R','N','D','C','Q','E','G','H','I',
    'L','K','M','F','P','S','T','W','Y','V']

standard_residues_three = [
    'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']

max_asa = {
    'ALA':129.0,'ARG':274.0,'ASN':195.0,'ASP':193.0,'CYS':167.0,
    'GLN':225.0,'GLU':223.0,'GLY':104.0,'HIS':224.0,'ILE':197.0,
    'LEU':201.0,'LYS':236.0,'MET':224.0,'PHE':240.0,'PRO':159.0,
    'SER':155.0,'THR':172.0,'TRP':285.0,'TYR':263.0,'VAL':174.0}

def format_data(pdb_paths: list, out_path: str):
    #cifp = MMCIFParser()
    cifp = PDBParser(QUIET=True)
    sasa = ShrakeRupley()
 
    dataset = {}
    for path in pdb_paths:
        print (path)
        pdb = path.split('/')[-1][0:4]
        structure = cifp.get_structure(pdb, path)

        dataset[pdb] = {
            'chains':{},
            'id_mask':{},
            'surface_mask':{},
            'paths':path}

        ## uniq 90% id chains would be better ...
        for chain in unfold_entities(structure, 'C'):
            id_mask = []
            chain_cloud = []
            surface_mask = []
            cid = chain.get_id()
            sasa.compute(chain, level='R')
            sasa.compute(chain, level='A')
            for residue in chain: 
                resname = residue.get_resname()
                for atom in residue:
                    if atom.get_id()[0] == 'H': continue
                    x, y, z = atom.get_coord()
                    chain_cloud.append([x, y, z])
                
                    if residue.sasa / max_asa[resname] >= 0.2 \
                    and atom.sasa != 0.0: surface_mask.append(1)
                    else: surface_mask.append(0)

                    id_mask.append(atom.get_full_id())
                    
            dataset[pdb]['chains'][cid] = jnp.array(chain_cloud, dtype=jnp.float32)
            dataset[pdb]['id_mask'][cid] = id_mask
            dataset[pdb]['surface_mask'][cid] = surface_mask

    dataset[pdb]['chains'] = initialize_clouds(dataset[pdb]['chains'], 42)

    with open(out_path, 'wb') as out:
        pickle.dump(dataset, out)


if __name__ == '__main__':
    datapath = '/proj/berzelius-2021-29/users/x_gabpo/complex_assembly/data/db5/*_r_u_cleaned.pdb'
    path_list = [line.rstrip() for line in glob.glob(datapath)]
    format_data(path_list, '/proj/berzelius-2021-29/users/x_gabpo/complex_assembly/data/train_features.pkl')
