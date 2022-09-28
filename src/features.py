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

from Bio import pairwise2
from Bio.Align import substitution_matrices
blosum62 = substitution_matrices.load("BLOSUM62")

from coordinates import *
from tables import *
from ops import *

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(figsize=(100, 100), dpi=100)
import seaborn as sb

def format_data(code, str1, str2):
    structures = [str1, str2]
    masks_pair, nodes_pair, cloud_pair = [], [], [] 
    for struc in structures:    
        sasa = ShrakeRupley()
        try:
            sasa.compute(struc, level='R')
            sasa.compute(struc, level='A')
        except: return [code, None]
        struc = unfold_entities(struc, 'R')
        
        masks, nodes, cloud = [], [], []
        for residue in struc: 
            rid = residue.get_resname()
            if rid not in standard_residues_three: continue

            # skip residues without CA
            if 'CA' not in residue: continue

            # get residue RSA
            masks.append(residue.sasa/max_asa[rid])

            # get CA coordinates to compute edges
            x, y, z = residue['CA'].get_coord()
            cloud.append([x, y, z])

            # get atom types/ASA to compute node features
            atoms = []
            for atom in residue:
                aid = atom.get_id()
                atoms.append(atom_types[rid][aid]+[atom.sasa])
            while len(atoms) < 16: atoms.append(12*[0])
            nodes.append(atoms)
            
        # store point cloud and surface mask
        masks_pair.append(jnp.array(masks))
        nodes_pair.append(jnp.array(nodes))
        cloud_pair.append(jnp.array(cloud))
    
    if len(cloud_pair[0]) == 0 or len(cloud_pair[1]) == 0: 
        print (f'Finished {code}')
        return [code, None]

    # always have largest protein first
    if len(cloud_pair[1]) > len(cloud_pair[0]):
        masks_pair = masks_pair[::-1]
        nodes_pair = nodes_pair[::-1]
        cloud_pair = cloud_pair[::-1]

    # compute cmaps as edge features and bin-encode them
    cloud1, cloud2 = cloud_pair    
    
    cmap1 = cmap_from_cloud(cloud1[:,None,:], cloud1[None,:,:])
    cmap2 = cmap_from_cloud(cloud2[:,None,:], cloud2[None,:,:])
    edges_pair = [
        one_hot_cmap(cmap1[:,:,None]), 
        one_hot_cmap(cmap2[:,:,None])]

    # compute interface true cmap
    icmap = cmap_from_cloud(cloud1[:,None,:], cloud2[None,:,:])
        
    # save original clouds
    cloud_pair_real = [cloud1, cloud2]

    # randomly rotate and center input clouds CoM to the origin
    cloud_pair = [
        initialize_clouds(cloud1, 42)[0], 
        initialize_clouds(cloud2, 42)[0]]

    #print (nodes_pair[0].shape, nodes_pair[1].shape, 
    #       cloud_pair[0].shape, cloud_pair[1].shape,
    #       edges_pair[0].shape, edges_pair[1].shape,
    #       masks_pair[0].shape, masks_pair[1].shape,
    #       cloud_pair_real[0].shape, cloud_pair_real[1].shape,
    #       icmap.shape)

    print (f'Finished {code}')
    return [code, {'nodes':nodes_pair, 'clouds':cloud_pair,
                 'edges':edges_pair, 'masks':masks_pair,
                 'interface':icmap, 'orig_clouds':cloud_pair_real}]
    

def plot_dataset_stats(dataset, subset_paths):
    data = {'pairsize':[], 'rec.size':[], 'lig.size':[], 
            'icontacts4':[], 'icontacts6':[], 'icontacts8':[], 
            'icontacts10':[], 'set':[]}

    for subset_path in subset_paths:
        subset_id = subset_path.split('/')[-2]
        paths = [line.rstrip() for line in glob.glob(subset_path)]
        codes = [path.split('/')[-1].strip('_r_b.pdb').upper() for path in paths]
        for code in codes:
            if code not in dataset: continue
            data['rec.size'].append(dataset[code]['clouds'][0].shape[0])
            data['lig.size'].append(dataset[code]['clouds'][1].shape[0])
            data['pairsize'].append(
                dataset[code]['clouds'][0].shape[0]+
                dataset[code]['clouds'][1].shape[0])
            data['icontacts4'].append(int(jnp.sum(jnp.where(dataset[code]['interface']<=4, 1, 0))))
            data['icontacts6'].append(int(jnp.sum(jnp.where(dataset[code]['interface']<=6, 1, 0))))
            data['icontacts8'].append(int(jnp.sum(jnp.where(dataset[code]['interface']<=8, 1, 0))))
            data['icontacts10'].append(int(jnp.sum(jnp.where(dataset[code]['interface']<=10, 1, 0))))
            data['set'].append(subset_id)

    f, ax = plt.subplots(3,2)
    data = pd.DataFrame(data)
    sb.kdeplot(x='rec.size', y='lig.size', data=data, hue='set', ax=ax[0][0])
    sb.histplot(x='pairsize', data=data, hue='set', fill=False, ax=ax[0][1])
    sb.histplot(x='icontacts6', data=data, hue='set', fill=False, ax=ax[1][1])
    sb.histplot(x='icontacts8', data=data, hue='set', fill=False, ax=ax[2][0])
    sb.histplot(x='icontacts10', data=data, hue='set', fill=False, ax=ax[2][1])
    f.legend(prop={'size': 0.5})
    plt.savefig('datasets.png')

def main():
    pdbp = PDBParser(QUIET=True)
    cifp = MMCIFParser(QUIET=True)

    b5_path = '/home/pozzati/complex_assembly/data/benchmark5.5/*_r_b.pdb'
    neg_path = '/home/pozzati/complex_assembly/data/negatome/*_r_b.pdb'
    output_path = '/home/pozzati/complex_assembly/data/train_features.pkl'
    path_list = [line.rstrip() for line in glob.glob(b5_path)]
    path_list += [line.rstrip() for line in glob.glob(neg_path)]
    
    
    rec_list, lig_list, code_list = [], [], []
    print ('Loading structures...')
    for path in path_list:
        path2 = path[:-7]+'l'+path[-6:]
        code_list.append(path.split('/')[-1].strip('_r_b.pdb').upper())
        rec_list.append(pdbp.get_structure('', path))
        lig_list.append(pdbp.get_structure('', path2))
  
    print ('Formatting...')
    formatted_list = jax.tree_util.tree_map(format_data, 
        code_list, rec_list, lig_list)

    dataset = {}
    for code, formatted_data in formatted_list:
        if formatted_data: dataset[code] = formatted_data
    
    with open(output_path, 'wb') as out:
        pickle.dump(dataset, out)

    plot_dataset_stats(dataset, [b5_path, neg_path])

if __name__ == '__main__':
    main()
