import os
import sys
import math
import copy
import json
import glob
import errno
import signal
import pickle
import string
import numpy as np

from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure
from Bio.PDB.Selection import unfold_entities

from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.SeqIO.PdbIO import PdbSeqresIterator
from Bio.SeqIO.PdbIO import CifSeqresIterator

from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Polypeptide import three_to_one

from tmtools import tm_align

standard_residues = [
    'A','R','N','D','C','Q','E','G','H','I',
    'L','K','M','F','P','S','T','W','Y','V']

standard_residues_three = [
    'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']



class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            except:
                pass
            finally:
                signal.alarm(0)
        return wrapper
    return decorator



@timeout(60)
def load_structure(path, parser):
    return parser.get_structure('', path)



def is_chain_valid(chain):

    polypep = False
    for residue in chain:
        if is_aa(residue): 
            polypep = True
            break

    for residue in chain:
        if residue.get_id()[0] == ' ' \
        and residue.get_resname() not in standard_residues_three: 
            return False

    if polypep: return True
    else: return False



def get_seq_coord(structure):
    sequence = ''
    coordinates = []
    residues = unfold_entities(structure, 'R')
    for residue in residues:
        if residue.get_id()[0] != ' ': continue
        if not 'CA' in residue: continue
        if residue.get_resname() in standard_residues_three:
            sequence += three_to_one(residue.get_resname())
            coordinates.append(list(residue['CA'].get_coord()))
    
    return sequence, np.array(coordinates)



def main():
#### load config file
    path = os.path.abspath(__file__)
    src_path = '/'.join(path.split('/')[:-1])
    config_path = src_path+'/config.json'
    with open(config_path,'r') as config:
        config = json.loads(config.read())

#### init classes
    pdbp = PDBParser(QUIET=True)
    cifp = MMCIFParser(QUIET=True)
    sasa = ShrakeRupley()

#### set up final dictionary or load pre-existing one
    pdb_dict = {'monomers':{}, 'noprot':{}, 'redundant':{}}
    write_path = config['base']+config['seqres']+'.pkl'
    if os.path.exists(write_path):
        with open(write_path, 'rb') as f:
            pdb_dict = pickle.load(f)

#### cycle over assemblies
    count = 0
    with open(config['base']+'/data/complex_list') as pdblist:
        pdblist = [line.strip() for line in pdblist]

    pathlist = []
    rawpaths = list(glob.glob(config['pdb']+'*/*'))
    for path_numb, path in enumerate(rawpaths):
        ######## print progression
        actual = int((float(path_numb+1)/len(rawpaths))*100)
        progression = '# {}% done! '.format(actual)
        print (progression, path)

        pdb_code = path.split('/')[-1][:4]
        if pdb_code in pdblist: pathlist.append(path)

    for path_numb, str_path in enumerate(pathlist):
       
######## print progression
        actual = int((float(path_numb+1)/len(pathlist))*100)
        progression = '# {}% done! '.format(actual)

        pdb_code = str_path.split('/')[-1][:4]
        suffix = str_path.split('.')[-1]

######## skip already done stuff
        if str_path in pdb_dict['noprot'].keys(): continue
        if str_path in pdb_dict['monomers'].keys(): continue
        if str_path in pdb_dict['redundant'].keys(): continue
        if pdb_code in pdb_dict and pdb_dict[pdb_code]['full_path'] == str_path: continue

######## get seqres info from file header
        with open(str_path) as handle:
            if 'pdb' in suffix:
                try:
                    seqres = {record.id:str(record.seq)
                        for record in PdbSeqresIterator(handle)}
                except: continue
            if 'cif' in suffix:
                try:
                    seqres = {record.id:str(record.seq)
                        for record in CifSeqresIterator(handle)}
                except: continue
        valid_assembly = True

######## load structure
        if 'pdb' in suffix:
            struc = load_structure(str_path, pdbp)
        if 'cif' in suffix:
            struc = load_structure(str_path, cifp)
        if not struc: continue
        
        chain_num = 0
        str_chains = []
        for model in struc:
            for chain in model:
                if is_chain_valid(chain):
                    chain_num += 1
                    str_chains.append(chain.get_id())

######## check identical PDB paths have not been processed yet
        if pdb_code in pdb_dict:
            if pdb_dict[pdb_code]['chain_num'] >= chain_num:
                 pdb_dict['redundant'][str_path] = 1
                 valid_assembly = False
                 fault = 'Processed equivalent!'

######## check seqres is not containing DNA/RNA
        if valid_assembly:
            for idx, seq in seqres.items():
                vocabulary = set([letter for letter in seq \
                    if letter in standard_residues or letter == 'U'])
                if vocabulary == {'A', 'T', 'C', 'G'} \
                or vocabulary == {'A', 'U', 'C', 'G'}:
                    pdb_dict['noprot'][str_path] = 1
                    valid_assembly = False
                    fault = 'Contains DNA/RNA!'
                    break

######## check PDB have no sequences with unknown residues
        if valid_assembly:
            unknown_residues = False
            for key, seq in seqres.items():
                for residue in seq:
                    if residue not in standard_residues:
                        unknown_residues = True
            if unknown_residues: 
                pdb_dict['noprot'][str_path] = 1
                valid_assembly = False
                fault = 'Unknown residues!'

######## check structure contains multiple chains
        if valid_assembly:
            seq_chains = list(seqres.keys())
            for chain in seq_chains:
                if chain not in str_chains: del(seqres[chain])

            if chain_num <= 1:
                pdb_dict['monomers'][str_path] = 1
                valid_assembly = False
                fault = 'Only 1 chain!'

        if valid_assembly:
            print (progression, str_path)
        else:
            print (progression, str_path, '### Discarded!', fault)
            continue

        pdb_dict[pdb_code] = {}
        pdb_dict[pdb_code]['full_path'] = str_path
        pdb_dict[pdb_code]['chain_num'] = chain_num
        pdb_dict[pdb_code]['seqres'] = copy.deepcopy(seqres)

        count += 1
        if count % 1000 == 0:
            with open(write_path, 'wb') as f:
                pickle.dump(pdb_dict, f)

    with open(write_path, 'wb') as f:
        pickle.dump(pdb_dict, f)


#        #compute SASA
#        cid = chains[0].get_id()
#        sasa.compute(struc, level='C')
#        print (struc[0][cid].sasa)        
#        for chain in chains:
#            sasa.compute(chain, level='C')
#            print (chain.sasa)
#            break
#        break
        

if __name__ == '__main__':

    main()

