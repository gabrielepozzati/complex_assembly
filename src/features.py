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

from coordinates import *
from tables import *
from graph import *
from ops import *

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(figsize=(100, 100), dpi=100)
import seaborn as sb

sasa = ShrakeRupley()

def pad(array, axes=1, size=1000):
    pad_size = size-array.shape[0]
    other_axes = len(array.shape)-axes
    if other_axes > 0: pad_shape = (((0,pad_size))*axes, ((0,0))*other_axes)
    else: pad_shape = ((0,pad_size))*axes
    return jnp.pad(array, pad_shape)

def filter_surface(array, surf_mask):
    return jnp.stack([line for line, asa in zip(array, surf_mask) if asa>=0.2], axis=0)

def format_data(code, str1, str2, key, enum=4, pad=1000):
    s = time.time()
    structures = [str1, str2]
    masks_pair, nodes_pair, cloud_pair = [], [], [] 
    for idx, struc in enumerate(structures):    
        try:
            sasa.compute(struc, level='R')
            sasa.compute(struc, level='A')
        except:
            print (f'Failed SASA computation in {code}')
            return [code, None]
        struc = unfold_entities(struc, 'R')
        
        masks, nodes, cloud = [], [], []
        for residue in struc: 
            rid = residue.get_resname()
            if rid not in standard_residues_three: continue

            # skip residues without CA
            if 'CA' not in residue: continue

            # get residue RSA
            masks.append(max(residue.sasa/max_asa[rid], 1e-3))

            # get CA coordinates to compute edges
            x, y, z = residue['CA'].get_coord()
            cloud.append([x, y, z])

            # get atom types/ASA to compute node features
            atoms = []
            for atom in standard_atoms:
                if atom not in residue: atoms.append(0)
                else: atoms.append(max(residue[atom].sasa, 1e-3))
            if idx == 0: nodes.append(atoms+[1,0])
            else: nodes.append(atoms+[0,1])
        
        while len(nodes) < pad: 
            nodes.append([0]*(len(standard_atoms)+2))

        if len(masks) > pad: 
            print (f'Sequence too long!')
            return [code, None]
        
        # store point cloud and surface mask
        masks_pair.append(jnp.array(masks))
        nodes_pair.append(jnp.array(nodes))
        cloud_pair.append(jnp.array(cloud))
    
    if len(cloud_pair[0]) == 0 or len(cloud_pair[1]) == 0: 
        print (f'One empty struct. in {code}')
        return [code, None]

    print ('Collect', time.time()-s)

    # compute cmaps as edge features and bin-encode them
    #mask1 = jnp.array([0.3,0.4,0.6,0.3])
    #mask2 = jnp.array([0.3,0.4,0.1])
    #cloud1 = jnp.array([[1,0,0],[2,0,0],[10,0,0],[50,0,0]])
    #cloud2 = jnp.array([[25,0,0],[100,0,0],[15,0,0]])

    s = time.time()
    cloud1, cloud2 = cloud_pair
    cmap1 = distances_from_cloud(cloud1[:,None,:], cloud1[None,:,:])
    cmap2 = distances_from_cloud(cloud2[:,None,:], cloud2[None,:,:])
    icmap = distances_from_cloud(cloud1[:,None,:], cloud2[None,:,:])
    print ('Cmaps', time.time()-s)

    s = time.time()
    edges1, senders1, receivers1 = get_edges(cmap1, enum)
    edges2, senders2, receivers2 = get_edges(cmap2, enum)
    senders2, receivers2 = senders2+pad, receivers2+pad

    padlen1 = (pad-len(cloud1))*enum
    padlen2 = (pad-len(cloud2))*enum
    
    edges1 = jnp.pad(edges1, (0, padlen1))
    edges2 = jnp.pad(edges2, (0, padlen2))
    senders1 = jnp.pad(senders1, (0, padlen1), constant_values=len(cloud1))
    senders2 = jnp.pad(senders2, (0, padlen2), constant_values=len(cloud2)+pad)
    receivers1 = jnp.pad(receivers1, (0, padlen1), constant_values=len(cloud1))
    receivers2 = jnp.pad(receivers2, (0, padlen2), constant_values=len(cloud2)+pad)

    edges12, senders12, receivers12,\
    edges21, senders21, receivers21 = get_interface_edges(icmap, enum, pad) 

    #print (cloud1)
    #print (cloud2)
    #print (mask1)
    #print (edges1.shape)
    #print (senders1.shape)
    #print (receivers1.shape)
    #print (icmap)
    #print (edges12)
    #print (senders12)
    #print (receivers12)
    #print (edges21)
    #print (senders21)
    #print (receivers21)
    #sys.exit()

    edges1 = encode_distances(edges1)
    edges2 = encode_distances(edges2)
    edges12 = encode_distances(edges12)
    edges21 = encode_distances(edges21)
    print ('Distances', time.time()-s)
    
    # randomly rotate and center input clouds CoM to the origin
    s = time.time()
    key1, key2 = jax.random.split(key)
    cloud_pair = [
        initialize_clouds(cloud1, key1), 
        initialize_clouds(cloud2, key2)]

    mask1, mask2 = masks_pair
    initial_transforms = [cloud_pair[0][1], cloud_pair[1][1]]
    
    mask1 = jnp.squeeze(jnp.where(mask1!=0, 1, 0))
    mask2 = jnp.squeeze(jnp.where(mask2!=0, 1, 0))
    cloud_pair = [cloud_pair[0][0]*mask1[:,None],
                  cloud_pair[1][0]*mask2[:,None]]
    print ('Init clouds', time.time()-s)


    print (f'Finished {code}')
    return [code, {'nodes':nodes_pair,
                   'edges':[edges1, edges2],
                   'iedges':[edges12, edges21],
                   'senders':[senders1, senders2],
                   'isenders':[senders12, senders21],
                   'receivers':[receivers1, receivers2],
                   'ireceivers':[receivers12, receivers21],
                   'clouds':cloud_pair, 'masks':masks_pair,
                   'init_rt':initial_transforms}]
    

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
        formatted_data = format_data(code, rec, lig, key1)

        if not formatted_data[1]: continue
        if path in neg_path_list:
            contacts = jnp.where(formatted_data[1]['interface']<=8, 1, 0)
            contacts = jnp.sum(contacts)
            if contacts > 0: continue
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
