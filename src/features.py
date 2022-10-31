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

def format_data(code, str1, str2):
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
            masks.append(residue.sasa/max_asa[rid])

            # get CA coordinates to compute edges
            x, y, z = residue['CA'].get_coord()
            cloud.append([x, y, z])

            # get atom types/ASA to compute node features
            atoms = []
            for atom in residue:
                aid = atom.get_id()
                if aid not in atom_types[rid]: continue
                atoms += atom_types[rid][aid]+[atom.sasa]
            while len(atoms) < 16: atoms += 12*[0]
            if idx == 0: nodes += atoms + [1,0]
            else: nodes += atoms + [0,1]
        
        if len(masks) > 1000: 
            print (f'Sequence too long!')
            return [code, None]
        
        # store point cloud and surface mask
        masks_pair.append(jnp.array(masks))
        nodes_pair.append(jnp.array(nodes))
        cloud_pair.append(jnp.array(cloud))
    
    if len(cloud_pair[0]) == 0 or len(cloud_pair[1]) == 0: 
        print (f'One empty struct. in {code}')
        return [code, None]

    # always have largest protein first
    if len(cloud_pair[1]) > len(cloud_pair[0]):
        masks_pair = masks_pair[::-1]
        nodes_pair = nodes_pair[::-1]
        cloud_pair = cloud_pair[::-1]
    print ('Collect', time.time()-s)

    # compute cmaps as edge features and bin-encode them
    mask1, mask2 = masks_pair
    cloud1, cloud2 = cloud_pair
    #mask1 = jnp.array([0.1,0.3,0.4,0.1,0.6,0.3,0.4,0.4])
    #cloud1 = jnp.array([[1,2,3],[1,0,0],[2,0,0],[1,2,3],[10,0,0],[50,0,0],[25,0,0],[100,0,0]])

    s = time.time()
    surf_cloud1 = filter_surface(cloud1, mask1)
    surf_cloud2 = filter_surface(cloud2, mask2)
    idx_map1 = jnp.squeeze(jnp.argwhere(mask1>=0.2))
    idx_map2 = jnp.squeeze(jnp.argwhere(mask2>=0.2))
    bidx_map1 = jnp.squeeze(jnp.argwhere(mask1<0.2))
    bidx_map2 = jnp.squeeze(jnp.argwhere(mask2<0.2))
    smasks_pair = [idx_map1, idx_map2]
    bmasks_pair = [bidx_map1, bidx_map2]
    print ('Split', time.time()-s)

    s = time.time()
    surf_cmap1 = cmap_from_cloud(surf_cloud1[:,None,:], surf_cloud1[None,:,:])
    surf_cmap2 = cmap_from_cloud(surf_cloud2[:,None,:], surf_cloud2[None,:,:])
    print ('Cmaps', time.time()-s)

    s = time.time()
    edges1, senders1, receivers1 = surface_edges(surf_cmap1, idx_map1)
    edges2, senders2, receivers2 = surface_edges(surf_cmap2, idx_map2)
    print ('Edges', time.time()-s)

    #print (cloud1)
    #print (mask1)
    #print (idx_map1)
    #print (surf_cmap1)
    #print (edges1, edges1.shape)
    #print (senders1, senders1.shape)
    #print (receivers1, receivers1.shape)

    s = time.time()
    edges1 = encode_distances(edges1)
    edges2 = encode_distances(edges2)
    print ('Distances', time.time()-s)

    s = time.time()
    graph_pair = [
            GraphsTuple(nodes=nodes_pair[0], edges=edges1,
                        senders=jnp.indices((10,)), receivers=receivers1,
                        n_node=jnp.array([nodes_pair[0].shape[0]]),
                        n_edge=jnp.array([edges1.shape[0]]),
                        globals=jnp.array([1,0])),
            GraphsTuple(nodes=nodes_pair[1], edges=edges2,
                        senders=senders2, receivers=receivers2,
                        n_node=jnp.array([nodes_pair[1].shape[0]]),
                        n_edge=jnp.array([edges2.shape[0]]),
                        globals=jnp.array([0,1]))]
    print ('Graph', time.time()-s)

    s = time.time()
    # compute interface true cmap
    icmap = cmap_from_cloud(cloud1[:,None,:], cloud2[None,:,:])
    print ('Interface labels', time.time()-s)

    # save original clouds
    cloud_pair_real = [cloud1, cloud2]

    s = time.time()
    # randomly rotate and center input clouds CoM to the origin
    cloud_pair = [
        initialize_clouds(cloud1, 42)[0], 
        initialize_clouds(cloud2, 42)[0]]
    print ('Init clouds', time.time()-s)

    print (f'Finished {code}')
    return [code, {'clouds':cloud_pair, 'orig_clouds':cloud_pair_real,
                   'smasks':smasks_pair, 'bmasks':bmasks_pair,
                   'graphs':graph_pair, 'interface':icmap}]
    

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

    for path in path_list:
        path2 = path[:-7]+'l'+path[-6:]
        code = path.split('/')[-1].strip('_r_b.pdb').upper()
        print ('loading')
        rec = pdbp.get_structure('', path)
        lig = pdbp.get_structure('', path2)
        print ('formatting')
        formatted_data = format_data(code, rec, lig)

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
