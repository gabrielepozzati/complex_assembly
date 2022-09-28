import os
import sys
import glob

from Bio.PDB.Selection import unfold_entities
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import PDBList

from Bio.PDB.PDBIO import Select
from Bio.PDB.PDBIO import PDBIO

class SelectChains(Select):
    def __init__(self, chain):
        self.chain = chain

    def accept_chain(self, chain):
        return chain.get_id() == self.chain

def download_pdbs(pdb_list, outdir):
    pdbl = PDBList()
    for pdb in pdb_list:
        pdbl.retrieve_pdb_file(pdb, pdir=outdir)

def main():
    io = PDBIO()
    pdbp = PDBParser(QUIET=True)
    cifp = MMCIFParser(QUIET=True)

    map_file = '/home/pozzati/complex_assembly/data/uniprot_pdb_map'
    list_file = '/home/pozzati/complex_assembly/data/negatome/neg_pdbstr_list'
    #list_file = '/home/pozzati/complex_assembly/test'
    output_path = '/home/pozzati/complex_assembly/data/negatome/'
   
    # correspondence negatome pdb : list of uniprot pairs
    negatome_dic = {}
    for line in open(list_file):
        pdb = line.rstrip().split()[2].upper()
        uni1 = line.rstrip().split()[0]
        uni2 = line.rstrip().split()[1]
        if pdb not in negatome_dic: negatome_dic[pdb] = []
        if set([uni1, uni2]) in negatome_dic[pdb]: continue
        else: negatome_dic[pdb].append(set([uni1, uni2]))

    # all uniprots in negatome
    negatome_uni = set(
        [line.rstrip().split()[0] for line in open(list_file)]+
        [line.rstrip().split()[1] for line in open(list_file)])

    # all pdbs in negatome
    negatome_pdb = set(negatome_dic.keys())

    print ('uni', len(negatome_uni), list(negatome_uni)[:10])
    print ('pdb', len(negatome_pdb), list(negatome_pdb)[:10])
    pairs = sum([len(negatome_dic[pdb]) for pdb in negatome_dic])
    print ('pairs', pairs, negatome_dic[list(negatome_pdb)[0]])
    
    # download negatome pdbs
    #download_pdbs(negatome_pdb, output_path)

    # map uniprot IDs to PDB chains
    uni_pdb_map = {}
    for line in open(map_file):
        
        # skip uniprot ids not present in negatome
        skip = True
        uni_codes = []
        uni_group = line.split('\t')[0]
        for uni in uni_group.split(';'):
            if uni.strip() in negatome_uni:
                uni_codes.append(uni.strip())
                skip = False
        if skip: continue

        # skip PDBs not present in negatome
        pdb_chain = None
        pdb_group = line.split('\t')[1:]
        for pdb in pdb_group:
            if pdb.strip() == '': continue
            
            skip = True
            code = pdb.split(';')[0].strip().upper()
            chain = pdb.split(';')[-1].strip()[0]
            if code in negatome_pdb: skip = False
            if skip: continue
            pdb_chain = chain
            break

        # store uniprot:chain correspondence
        if pdb_chain: 
            for uni_code in uni_codes: uni_pdb_map[uni_code] = pdb_chain
        else: continue

    # for each downloaded negatome PDB
    path_list = [line.rstrip() for line in glob.glob(output_path+'*.cif')]
    for path in path_list:
        pdb = path.split('/')[-1][:4].upper()
        struc = cifp.get_structure('', path)
        io.set_structure(struc)

        # for each negatome pair of chains in that PDB
        for uni1, uni2 in negatome_dic[pdb]:
            out_complex = f'{uni1}_{uni2}'

            # if a uniprot:PDBchain mapping exist for both uniprots
            if uni1 not in uni_pdb_map \
            or uni2 not in uni_pdb_map:
                print (uni1, uni2, 'pair not found!')
                continue

            # try to extract and save mapped chains 
            chain_r = uni_pdb_map[uni1]
            chain_l = uni_pdb_map[uni2]
            io.save(output_path+out_complex+'_r_b.pdb', SelectChains(chain_r))
            io.save(output_path+out_complex+'_l_b.pdb', SelectChains(chain_l))


if __name__ == '__main__':
    main()

