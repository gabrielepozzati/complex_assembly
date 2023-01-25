import os
import sys
import glob
import time
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

#def pad(array, axes=1, size=1000):
#    pad_size = size-array.shape[0]
#    other_axes = len(array.shape)-axes
#    if other_axes > 0: pad_shape = (((0,pad_size))*axes, ((0,0))*other_axes)
#    else: pad_shape = ((0,pad_size))*axes
#    return jnp.pad(array, pad_shape)

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

#    # compute distance maps
#    cloud1, cloud2 = CA_pair
#    cmap1 = distances_from_cloud(cloud1[:,None,:], cloud1[None,:,:])
#    cmap2 = distances_from_cloud(cloud2[:,None,:], cloud2[None,:,:])
#
#    # compute, pad and encode intra-chain edges
#    local1, local2 = local_pair
#    edges1, senders1, receivers1 = get_edges(cmap1, local1, local1, cloud1, cloud1, enum)
#    edges2, senders2, receivers2 = get_edges(cmap2, local2, local2, cloud2, cloud2, enum)
#
#    # compute , pad and encode ground truth inter-chain edges 
#    icmap = distances_from_cloud(cloud1[:,None,:], cloud2[None,:,:])
#    ledges12, lsenders12, lreceivers12 = get_edges(icmap, local1, local2, cloud1, cloud2, enum)
#    
#    icmap = jnp.transpose(icmap)
#    ledges21, lsenders21, lreceivers21 = get_edges(icmap, local2, local1, cloud2, cloud1, enum) 
#
#    ledges = jnp.concatenate((ledges12, ledges21), axis=0)
#    lsenders = jnp.concatenate((lsenders12, lsenders21), axis=0)
#    lreceivers = jnp.concatenate((lreceivers12, lreceivers21), axis=0)
#
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
    

def plot_dataset_stats(dataset, subset_paths):
    data = {'pairsize':[], 'rec.size':[], 'lig.size':[], 
            'icontacts6':[], 'icontacts8':[], 'set':[]}

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
            data['icontacts6'].append(int(jnp.sum(jnp.where(dataset[code]['interface']<=6, 1, 0))))
            data['icontacts8'].append(int(jnp.sum(jnp.where(dataset[code]['interface']<=8, 1, 0))))
            data['set'].append(subset_id)

    f, ax = plt.subplots(2,2)
    f.tight_layout(pad=2.0)
    data = pd.DataFrame(data)
    sb.kdeplot(x='rec.size', y='lig.size', data=data, hue='set', legend=False, ax=ax[0][0])
    sb.histplot(x='pairsize', data=data, hue='set', fill=False, ax=ax[0][1])
    sb.scatterplot(x='pairsize', y='icontacts6', data=data, hue='set', s=3, legend=False, ax=ax[1][0])
    sb.scatterplot(x='pairsize', y='icontacts8', data=data, hue='set', s=3, legend=False, ax=ax[1][1])
    ax[1][0].set_ylim([0, 200])
    ax[1][1].set_ylim([0, 200])
    plt.savefig('datasets.png')

def main():
    pdbp = PDBParser(QUIET=True)
    cifp = MMCIFParser(QUIET=True)

    b5_path = '/home/pozzati/complex_assembly/data/benchmark5.5/*_r_b.pdb'
    neg_path = '/home/pozzati/complex_assembly/data/negatome/*_r_b.pdb'
    output_path = '/home/pozzati/complex_assembly/data/dataset_features/'
    pos_path_list = [line.rstrip() for line in glob.glob(b5_path)]
    neg_path_list = [line.rstrip() for line in glob.glob(neg_path)]
    path_list = pos_path_list+neg_path_list

    key = jax.random.PRNGKey(46)

    for path in path_list:
        path2 = path[:-7]+'l'+path[-6:]
        code = path.split('/')[-1].strip('_r_b.pdb').upper()
        print ('loading')
        rec = pdbp.get_structure('', path)
        lig = pdbp.get_structure('', path2)
        print ('formatting')
        key, key1 = jax.random.split(key)
        formatted_data = format_data(code, rec, lig, key1, path in neg_path_list)

        if not formatted_data[1]: continue
        with open(output_path+code+'.pkl', 'wb') as out:
            pickle.dump(formatted_data[1], out)

def main2():
    b5_path = '/home/pozzati/complex_assembly/data/benchmark5.5/*_r_b.pdb'
    neg_path = '/home/pozzati/complex_assembly/data/negatome/*_r_b.pdb'
    output_path = '/home/pozzati/complex_assembly/data/train_features.pkl'
    with open(output_path, 'rb') as data:
        dataset = pickle.load(data)

    plot_dataset_stats(dataset, [b5_path, neg_path])

if __name__ == '__main__':
    main()
