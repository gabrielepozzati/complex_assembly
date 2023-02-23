import os
import sys
import glob
import time
import json
import pickle
import jax.numpy as jnp
import pdb as debug
from jraph import *

from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Polypeptide import three_to_one
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Selection import unfold_entities

from tables import *
from ops import *

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(figsize=(100, 100), dpi=100)
import seaborn as sb

sasa = ShrakeRupley()

#def filter_surface(array, surf_mask):
#    return jnp.stack([line for line, asa in zip(array, surf_mask) if asa>=0.2], axis=0)

def format_data(code, str1, str2, key, negative, enum=10, maxlen=400):
    structures = [str1, str2]

    masks_pair, nodes_pair, coord_pair = [], [], []
    for idx, struc in enumerate(structures):    
        try:
            sasa.compute(struc, level='R')
            sasa.compute(struc, level='A')
        except:
            print (f'Failed SASA computation in {code}')
            return [code, None]

        struc = unfold_entities(struc, 'R')
        
        masks, nodes, coord = [], [], []
        for residue in struc: 
            rid = residue.get_resname()
            if rid not in standard_residues_three: continue

            # skip residues without CA
            if 'CA' not in residue: continue

            # get residue RSA
            res_rsa = max(residue.sasa, 1)/max_asa[rid]
            
            # skip buried residues
            if res_rsa < 0.2: continue
            
            masks.append(res_rsa)

            # get residue 1-hot
            if idx == 0: nodes.append(residue_1hot[rid]+[1,0])
            else: nodes.append(residue_1hot[rid]+[0,1])

            # get CA coordinates to compute local frames and edges
            coord.append(jnp.array(residue['CA'].get_coord()))

        # store point cloud and surface mask
        masks_pair.append(jnp.array(masks))
        nodes_pair.append(jnp.array(nodes))
        coord_pair.append(jnp.array(coord))

    if len(coord_pair[0]) == 0 or len(coord_pair[1]) == 0 \
    or len(coord_pair[0]) >= maxlen or len(coord_pair[1]) >= maxlen: 
        print (f'One empty/too long struct. in {code}')
        return [code, None]

#    # sanity check on negatome examples not having contacts
#    if jnp.max(icmap) <= 8 and negative:
#        print (f'Existing interface in negative example: {code}')
#        return [code, None]

    # pad clouds and masks
    padlen1 = maxlen-len(nodes_pair[0])
    padlen2 = maxlen-len(nodes_pair[1])
    masks_pair = [
            jnp.concatenate((masks_pair[0], jnp.zeros((padlen1,))), axis=0),
            jnp.concatenate((masks_pair[1], jnp.zeros((padlen2,))), axis=0)]

    coord_pair = [
            jnp.concatenate((coord_pair[0], jnp.zeros((padlen1, 3))), axis=0),
            jnp.concatenate((coord_pair[1], jnp.zeros((padlen2, 3))), axis=0)]

    nodes_pair = [
            jnp.concatenate((nodes_pair[0], jnp.zeros((padlen1, 22))), axis=0),
            jnp.concatenate((nodes_pair[1], jnp.zeros((padlen2, 22))), axis=0)]

    print (f'Finished {code}')
    return [code, {'coord':coord_pair,
                   'masks':masks_pair,
                   'nodes':nodes_pair}]
    

def plot_dataset_stats(subset_paths, config):
    data = {'rec_size':[], 'lig_size':[], 
            'rec_size_acc':[], 'lig_size_acc':[]}

    for subset_path in subset_paths:
        for idx, path in enumerate(glob.glob(subset_path+'/*')):
            with open(path, 'rb') as f: feats = pkl.load(f)
            #subset_id = subset_path.split('/')[-2]
            rec_ca = jnp.argwhere(feats['masks'][0]!=0)
            lig_ca = jnp.argwhere(feats['masks'][1]!=0)
            rec_ca_acc = jnp.argwhere(feats['masks'][0]>=0.2)
            lig_ca_acc = jnp.argwhere(feats['masks'][1]>=0.2)
            data['rec_size'].append(rec_ca.shape[0])
            data['lig_size'].append(lig_ca.shape[0])
            data['rec_size_acc'].append(rec_ca_acc.shape[0])
            data['lig_size_acc'].append(lig_ca_acc.shape[0])
            #rec_coord = dataset[code]['clouds'][0][rec_ca] 
            #lig_coord = dataset[code]['clouds'][1][lig_ca]
            #dmap = distances_from_coords(rec_coord[:,None,:], lig_coord[None,:,:])
    
            #rec_coord_acc = dataset[code]['clouds'][0][rec_ca_acc]
            #lig_coord_acc = dataset[code]['clouds'][1][lig_ca_acc]
            #dmap_acc = distances_from_coords(rec_coord_acc[:,None,:], lig_coord_acc[None,:,:])
    
            #data['icontacts'].append(int(jnp.sum(jnp.where(dmap<=config['interface_threshold'], 1, 0))))
            #data['icontacts_acc'].append(int(jnp.sum(jnp.where(dmap<=config['interface_threshold'], 1, 0))))
            #data['set'].append(subset_id)

    f, ax = plt.subplots(2,1)
    f.tight_layout(pad=2.0)
    data = pd.DataFrame(data)
    sb.scatterplot(x='rec_size', y='lig_size', data=data, legend=False, ax=ax[0])
    sb.scatterplot(x='rec_size_acc', y='lig_size_acc', data=data, legend=False, ax=ax[1])
    #sb.scatterplot(x='pairsize', y='icontacts6', data=data, hue='set', s=3, legend=False, ax=ax[1][0])
    #sb.scatterplot(x='pairsize', y='icontacts8', data=data, hue='set', s=3, legend=False, ax=ax[1][1])
    #ax[1][0].set_ylim([0, 200])
    #ax[1][1].set_ylim([0, 200])
    plt.savefig('datasets.png')

def main():
    pdbp = PDBParser(QUIET=True)
    cifp = MMCIFParser(QUIET=True)

    path = os.getcwd()+'/'+'/'.join(__file__.split('/')[:-2])

    with open(path+'/src/config.json') as j: config = json.load(j)

    b5_path = f'{path}/data/benchmark5.5/*_r_b.pdb'
    neg_path = f'{path}/data/negatome/*_r_b.pdb'
    output_path = f'{path}/data/dataset_features/'
    pos_path_list = [line.rstrip() for line in glob.glob(b5_path)]
    neg_path_list = [line.rstrip() for line in glob.glob(neg_path)]
    path_list = pos_path_list #+neg_path_list

    key = jax.random.PRNGKey(config['random_seed'])

    for path in path_list:
        path2 = path[:-7]+'l'+path[-6:]
        code = path.split('/')[-1].strip('_r_b.pdb').upper()
        rec = pdbp.get_structure('', path)
        lig = pdbp.get_structure('', path2)
        key, key1 = jax.random.split(key)
        formatted_data = format_data(
                code, rec, lig, key1, 
                path in neg_path_list, 
                maxlen=config['padding'])

        if not formatted_data[1]: continue
        with open(output_path+code+'.pkl', 'wb') as out:
            pickle.dump(formatted_data[1], out)

def main2():
    
    path = os.getcwd()+'/'+'/'.join(__file__.split('/')[:-2])
    with open(path+'/src/config.json') as j: config = json.load(j)

    b5_path = f'{path}/data/dataset_features'
    #neg_path = f'{path}/data/negatome/*_r_b.pdb'

    plot_dataset_stats([b5_path], config)#, neg_path])

if __name__ == '__main__':
    main()
    #main2()
