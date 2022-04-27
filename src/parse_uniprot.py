import os
import json
import pickle

def main():

#### load config file
    path = os.path.abspath(__file__)
    src_path = '/'.join(path.split('/')[:-1])
    config_path = src_path+'/config.json'
    with open(config_path,'r') as config:
        config = json.loads(config.read())

    with open(config['uni']) as uniprot:
        entries = ''.join([line for line in uniprot])
    
    uni_mapping = {}
    entries = entries.split('//')
    for entry in entries:

        entry = entry.split('\n')
        for line in entry: 

            if line.startswith('AC   '):
                uni_codes = line[5:].strip(';').split(';')
                for index, code in enumerate(uni_codes):
                    uni_codes[index] = code.strip()
                   
            elif line.startswith('DR   PDB;'):
                pdb_code = line.split()[2].strip(';').lower()
                if pdb_code not in uni_mapping:
                    uni_mapping[pdb_code] = {}

                chains = line.split(';')[4]
                for chain_group in chains.split(','):
                    chain_group = chain_group.strip()
                    chain_group = chain_group.split('=')[0]
                    chain_group = chain_group.split('/')

                    for chain in chain_group:
                        if chain not in uni_mapping[pdb_code]:
                            uni_mapping[pdb_code][chain] = []
                        uni_mapping[pdb_code][chain] += uni_codes
                        
    print (uni_mapping)

    output = config['base']+'data/pdb_uni_map.pkl'
    with open(output, 'wb') as f:
            pickle.dump(uni_mapping, f)



if __name__ == '__main__':

    main()

