import os
import sys
import glob
import pickle
import jax.numpy as jnp
import pdb as debug


from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Polypeptide import three_to_one
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Selection import unfold_entities
from coordinates import *

import matplotlib.pyplot as plt


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

atom_types = {'ALA':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1]},
              'ARG':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'NE':[0,0,0,1,0,0,0,0,0,0,0],
                     'NH1':[0,0,0,1,0,0,0,0,0,0,0],
                     'NH2':[0,0,0,1,0,0,0,0,0,0,0],
                      'CZ':[0,0,0,0,0,0,0,0,1,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                      'CG':[0,0,0,0,0,0,0,0,0,0,1],
                      'CD':[0,0,0,0,0,0,0,0,0,0,1]},
              'ASN':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                     'ND2':[0,1,0,0,0,0,0,0,0,0,0],
                     'OD1':[0,0,0,0,0,1,0,0,0,0,0],
                      'CG':[0,0,0,0,0,0,0,0,1,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1]},
              'ASP':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                     'OD1':[0,0,0,0,0,0,0,1,0,0,0],
                     'OD2':[0,0,0,0,0,0,0,1,0,0,0],
                      'CG':[0,0,0,0,0,0,0,0,1,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1]},
              'CYS':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'SG':[1,0,0,0,0,0,0,0,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1]},
              'GLN':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                     'NE2':[0,1,0,0,0,0,0,0,0,0,0],
                     'OE1':[0,0,0,0,0,1,0,0,0,0,0],
                      'CD':[0,0,0,0,0,0,0,0,1,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                      'CG':[0,0,0,0,0,0,0,0,0,0,1]},
              'GLU':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                     'OE1':[0,0,0,0,0,0,0,1,0,0,0],
                     'OE2':[0,0,0,0,0,0,0,1,0,0,0],
                      'CD':[0,0,0,0,0,0,0,0,1,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                      'CG':[0,0,0,0,0,0,0,0,0,0,1]},
              'GLY':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0]},
              'HIS':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                     'ND1':[0,0,1,0,0,0,0,0,0,0,0],
                     'NE2':[0,0,1,0,0,0,0,0,0,0,0],
                      'CG':[0,0,0,0,0,0,0,0,0,1,0],
                     'CD2':[0,0,0,0,0,0,0,0,0,1,0],
                     'CE1':[0,0,0,0,0,0,0,0,0,1,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1]},
              'ILE':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                     'CG1':[0,0,0,0,0,0,0,0,0,0,1],
                     'CG2':[0,0,0,0,0,0,0,0,0,0,1],
                     'CD1':[0,0,0,0,0,0,0,0,0,0,1]},
              'LEU':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                      'CG':[0,0,0,0,0,0,0,0,0,0,1],
                     'CD1':[0,0,0,0,0,0,0,0,0,0,1],
                     'CD2':[0,0,0,0,0,0,0,0,0,0,1]},
              'LYS':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'NZ':[0,0,0,0,1,0,0,0,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                      'CG':[0,0,0,0,0,0,0,0,0,0,1],
                      'CD':[0,0,0,0,0,0,0,0,0,0,1],
                      'CE':[0,0,0,0,0,0,0,0,0,0,1]},
              'MET':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                      'CG':[0,0,0,0,0,0,0,0,0,0,1],
                      'CE':[0,0,0,0,0,0,0,0,0,0,1],
                      'SD':[1,0,0,0,0,0,0,0,0,0,0]},
              'PHE':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                      'CG':[0,0,0,0,0,0,0,0,0,1,0],
                     'CD1':[0,0,0,0,0,0,0,0,0,1,0],
                     'CD2':[0,0,0,0,0,0,0,0,0,1,0],
                     'CE1':[0,0,0,0,0,0,0,0,0,1,0],
                     'CE2':[0,0,0,0,0,0,0,0,0,1,0],
                      'CZ':[0,0,0,0,0,0,0,0,0,1,0]},
              'PRO':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                      'CG':[0,0,0,0,0,0,0,0,0,0,1],
                      'CD':[0,0,0,0,0,0,0,0,0,0,1]},
              'SER':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'OG':[0,0,0,0,0,0,1,0,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1]},
              'THR':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                     'OG1':[0,0,0,0,0,0,1,0,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                     'CG2':[0,0,0,0,0,0,0,0,0,0,1]},
              'TRP':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                      'CG':[0,0,0,0,0,0,0,0,0,1,0],
                     'CD1':[0,0,0,0,0,0,0,0,0,1,0],
                     'CD2':[0,0,0,0,0,0,0,0,0,1,0],
                     'CE1':[0,0,0,0,0,0,0,0,0,1,0],
                     'CE2':[0,0,0,0,0,0,0,0,0,1,0],
                     'CE3':[0,0,0,0,0,0,0,0,0,1,0],
                     'CZ2':[0,0,0,0,0,0,0,0,0,1,0],
                     'CZ3':[0,0,0,0,0,0,0,0,0,1,0],
                     'CH2':[0,0,0,0,0,0,0,0,0,1,0],
                     'NE1':[0,0,1,0,0,0,0,0,0,0,0]},
              'TYR':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'CG':[0,0,0,0,0,0,0,0,0,1,0],
                     'CD1':[0,0,0,0,0,0,0,0,0,1,0],
                     'CD2':[0,0,0,0,0,0,0,0,0,1,0],
                     'CE1':[0,0,0,0,0,0,0,0,0,1,0],
                     'CE2':[0,0,0,0,0,0,0,0,0,1,0],
                      'CZ':[0,0,0,0,0,0,0,0,0,1,0],
                      'OH':[0,0,0,0,0,0,1,0,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1]},
              'VAL':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                     'CG1':[0,0,0,0,0,0,0,0,0,0,1],
                     'CG2':[0,0,0,0,0,0,0,0,0,0,1]}}

def format_data(pdb_paths: list, out_path: str):
    #cifp = MMCIFParser()
    cifp = PDBParser(QUIET=True)
    sasa = ShrakeRupley()
 
    dataset = {}
    for path in pdb_paths:
        print (path)
        pdb = path.split('/')[-1][0:4]
        path2 = path[:-7]+'l'+path[-6:]
        structure1 = cifp.get_structure(pdb, path)
        structure2 = cifp.get_structure(pdb, path2)
        dataset[pdb] = {}

        cloud = []
        id_mask = []
        surf_mask = []
        surf_features = []
        for chain in unfold_entities(structure1, 'C')+unfold_entities(structure2, 'C'):
            sasa.compute(chain, level='R')
            sasa.compute(chain, level='A')
            cid = chain.get_id()
            
            for residue in chain: 
                rid = residue.get_resname()
                if rid not in standard_residues_three: continue

                # get CA coordinates to compute edges
                if 'CA' not in residue: continue
                x, y, z = residue['CA'].get_coord()
                cloud.append([x, y, z])

                # get residue RSA
                surf_mask.append(residue.sasa/max_asa[rid])

                # add some tracking to the original structure lines
                id_mask.append(residue.get_full_id())

                # get atom types/ASA to compute features
                surf_feature = []
                for atom in residue:
                    aid = atom.get_id()
                    surf_feature.append(atom_types[rid][aid]+[atom.sasa])
                while len(surf_feature) < 16: surf_feature.append(12*[0])
                surf_features.append(surf_feature)
        
        # compute all vs all cmap
        cloud = jnp.array(cloud)
        cloud_len = cloud.shape[0]
        cloud1 = jnp.expand_dims(cloud, axis=1)
        cloud2 = jnp.expand_dims(cloud, axis=0)
        cloud1 = jnp.broadcast_to(cloud1, (cloud_len, cloud_len, 3))
        cloud2 = jnp.broadcast_to(cloud2, (cloud_len, cloud_len, 3))
        cmap = jnp.sqrt(jnp.sum((cloud1-cloud2)**2, axis=-1))

        # compute edges and edges features
        sender, receiver = [], []
        edge_feats, edge_labels = [], []
        for idx1 in range(cloud_len):
            for idx2 in range(cloud_len):
                if idx1 == idx2: continue
                sender.append(idx1)
                receiver.append(idx2)
                edge_feats.append(cmap[idx1][idx2])

                cid1 = id_mask[idx1][2]
                cid2 = id_mask[idx2][2]
                if cid1 == cid2: edge_labels.append([1,0])
                else: edge_labels.append([0,1])

        cloud, tgroup = initialize_clouds(cloud, id_mask, 42)

        dataset[pdb]['cloud'] = jnp.array(cloud, dtype=jnp.float32)
        dataset[pdb]['nodes'] = jnp.array(surf_features, dtype=jnp.float32)
        dataset[pdb]['edges'] = jnp.array(edge_feats, dtype=jnp.float32)
        dataset[pdb]['sender'] = jnp.array(sender, dtype=jnp.float32)
        dataset[pdb]['receiver'] = jnp.array(receiver, dtype=jnp.float32)
        dataset[pdb]['labels'] = jnp.array(edge_labels, dtype=jnp.float32)
        dataset[pdb]['smask'] = jnp.array(surf_mask, dtype=jnp.float32)
        dataset[pdb]['imask'] = id_mask
        dataset[pdb]['tgroup'] = tgroup

        #print (tgroup)
        #print (id_mask)
        #in_paths = [path, path2]
        #write_mmcif(id_mask, tgroup, in_paths, '/proj/berzelius-2021-29/users/x_gabpo/complex_assembly/test.cif')
        #sys.exit()

    with open(out_path, 'wb') as out:
        pickle.dump(dataset, out)


if __name__ == '__main__':
    datapath = '/proj/berzelius-2021-29/users/x_gabpo/complex_assembly/data/benchmark5.5/*_r_b.pdb'
    path_list = [line.rstrip() for line in glob.glob(datapath)]
    format_data(path_list, '/proj/berzelius-2021-29/users/x_gabpo/complex_assembly/data/train_features.pkl')
