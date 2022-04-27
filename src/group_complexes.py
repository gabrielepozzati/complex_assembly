import os 
import json
import glob
import pickle
import pandas as pd
import subprocess as sp
import seaborn as sb
import matplotlib.pyplot as plt

def main():
#### load config file
    path = os.path.abspath(__file__)
    src_path = '/'.join(path.split('/')[:-1])
    config_path = src_path+'/config.json'
    with open(config_path,'r') as config:
        config = json.loads(config.read())

#### load pkled data
    seqres_path = config['base']+config['seqres']+'.pkl'
    with open(seqres_path, 'rb') as f:
        seqres = pickle.load(f)

    unimap_path = config['base']+'/data/pdb_uni_map.pkl'
    with open(unimap_path, 'rb') as f:
        pdbtouni = pickle.load(f)

#### for each protein in seqres
    write_path = config['base']+'data/complexes/'
    if not os.path.exists(write_path): os.mkdir(write_path)

    uniq_complexes = {}
    for pdb in seqres:
        
######## skip monomers
        if len(pdb) > 4: continue
        chain_number = seqres[pdb]['chain_num']
        if int(chain_number) <= 1: continue

######## skip empty seqres
        if len(seqres[pdb]['seqres']) == 0: continue
        
        uniprots = []
        for chain in seqres[pdb]['seqres']:
            if pdb in pdbtouni:
                if chain in pdbtouni[pdb]: 
                    uniprots.append(pdbtouni[pdb][chain][0])
        uniprots.sort()
        complex_tag = ''.join(uniprots)

        if complex_tag not in uniq_complexes:
            uniq_complexes[complex_tag] = 1
            command = 'cp {} {}'.format(
                seqres[pdb]['full_path'],
                write_path)
            
            sp.run(command, shell=True)
            print (pdb, 'done! Uniq complex found!')
        else:
            print (pdb, 'discarded!')

if __name__ == '__main__':

    main()



