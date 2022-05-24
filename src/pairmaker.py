import sys
import glob
import json
import itertools

def fasta_combinations(chain_number, af_out_folder):
    af_out_folder = af_out_folder.rstrip('/')
    mapping = af_out_folder+'/msas/chain_id_map.json'
    mapping = open(mapping)
    mapping_dict = json.load(mapping)

    chains = []
    for path in glob.glob(af_out_folder+'/msas/*'):
        print (path)
        if path.endswith('.json'): continue
        chains.append(path.split('/')[-1])

    pdb = af_out_folder.split('/')[-1]
    sets = itertools.combinations_with_replacement(chains, chain_number)
    for idx, chain_set in enumerate(sets):
        idx += 1
        with open(f'{af_out_folder}/{pdb}_{idx}.fasta', 'w') as out:
            for chain in chain_set:
                out.write('>'+mapping_dict[chain]['description']+'\n')
                out.write(mapping_dict[chain]['sequence']+'\n')

def main():
    fasta_combinations(2, sys.argv[1]) 

if __name__ == '__main__':
    main()
