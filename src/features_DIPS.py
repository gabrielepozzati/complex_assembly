import os
import sys
import glob
import jraph
import pandas as pd
import dill as pickle
import jax.numpy as jnp
import pdb as debug

from coordinates import *
from tables import *

import matplotlib.pyplot as plt

def get_inter_cmap(cloud1, cloud2):
    return jnp.sqrt(jnp.sum((cloud1[None,:,:]-cloud2[:,None,:])**2, axis=-1))

def format_data(pair_group):

    best_count, best_chains, best_int = 0, None, None
    for path in pair_group:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        chain1 = data[1][['x','y','z','atom_name','resname','rsa_value']]
        cloud1 = jnp.array(chain1[chain1['atom_name']=='CA'][['x','y','z']])
        chain2 = data[2][['x','y','z','atom_name','resname','rsa_value']]
        cloud2 = jnp.array(chain2[chain2['atom_name']=='CA'][['x','y','z']])
        inter_cmap = get_inter_cmap(cloud1, cloud2)
        contact_num = jnp.sum(jnp.where(inter_cmap<8, 1, 0))
        if contact_num >= best_count:
            best_count = contact_num
            best_int = inter_cmap
            if len(chain1) >= len(chain2): best_chains = [chain1, chain2]
            else: best_chains = [chain2, chain1]
    
    chain1 = best_chains[0]
    chain2 = best_chains[1]
    cloud1 = jnp.array(chain1[chain1['atom_name']=='CA'][['x','y','z']])
    cloud1, tgroup = initialize_clouds(cloud1, 42)
    cloud2 = jnp.array(chain2[chain2['atom_name']=='CA'][['x','y','z']])
    cloud2, tgroup = initialize_clouds(cloud2, 42)
    mask1 = jnp.array(pd.to_numeric(chain1['rsa_value'], errors='coerce').fillna(0.0))
    mask2 = jnp.array(pd.to_numeric(chain2['rsa_value'], errors='coerce').fillna(0.0))

    return {'masks' : [mask1, mask2], 'nodes' : [None, None], 
            'clouds' : [cloud1, cloud2], 'interface':best_int}
    

def main():
    
    data_path = '/home/pozzati/complex_assembly/data/raw/*/*'
    output_path = '/home/pozzati/complex_assembly/data/train_features.pkl'
    
    # reduce multiple PDBs with identical uniprot IDs sets
    pdb_uni_map = {}
    for line in open('/home/pozzati/complex_assembly/data/pdb_chain_uniprot.csv'):
        if line.startswith('# ') or line.startswith('PDB,'): continue
        pdb = line.split(',')[0]
        uni = line.split(',')[2]
        if pdb in pdb_uni_map: pdb_uni_map[pdb].add(uni)
        else: pdb_uni_map[pdb] = set(uni)

    path_list = [line.rstrip() for line in glob.glob(data_path)]
    pdb_list = list(set([path.split('/')[-1].split('.pdb')[0].lower() \
        for path in path_list]))
    unique_set = []
    unique_pdb = []
    for pdb in pdb_list:
        if pdb not in pdb_uni_map: continue
        if pdb_uni_map[pdb] not in unique_set:
            unique_pdb.append(pdb)
            unique_set.append(pdb_uni_map[pdb])
    print (len(unique_pdb),'unique PDB codes found!')

    # group pairs from the same PDB
    path_dict = {}
    for path in path_list:
        pdb = path.split('/')[-1].split('.pdb')[0].lower()
        if pdb not in unique_pdb: continue
        if pdb not in path_dict: path_dict[pdb] = [path]
        else: path_dict[pdb].append(path)
        
    # select largest interface pair and format
    if os.path.exists(output_path):
        with open(output_path, 'rb') as f:
            dataset = pickle.load(f)
    else: dataset = {}

    print (len(list(dataset.keys())),'already there!')
    list_keys = list(dataset.keys())
    for pdb in list_keys:
        if pdb not in path_dict: del(dataset[pdb])
        if pdb in path_dict: del(path_dict[pdb])
    print (len(list(dataset.keys())),'already there!')
    print (len(list(path_dict.keys())),'to go!')

    for idx, (pdb, pair_group) in enumerate(path_dict.items()):
        print (f'Formatting {pdb} chains..')
        dataset[pdb] = format_data(pair_group)
        if idx % 100 == 0:
            with open(output_path, 'wb') as out:
                pickle.dump(dataset, out)

    print (len(list(dataset.keys())), 'pairs formatted!')
    with open(output_path, 'wb') as out:
        pickle.dump(dataset, out)

if __name__ == '__main__':
    main()
