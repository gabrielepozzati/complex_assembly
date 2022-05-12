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

from Bio import pairwise2
from Bio.Align import substitution_matrices
blosum62 = substitution_matrices.load("BLOSUM62")

from tmtools import tm_align

standard_residues = [
    'A','R','N','D','C','Q','E','G','H','I',
    'L','K','M','F','P','S','T','W','Y','V']

standard_residues_three = [
    'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']

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



class DataCleaner():
    def __init__(self, pdblist, data_folder, tmp_folder):

        with open(pdblist) as list_file:
            self.pdblist = [line.strip() for line in list_file]

        self.data_folder = data_folder
        self.tmp_folder = tmp_folder

        self.pdbp = PDBParser(QUIET=True)
        self.cifp = MMCIFParser(QUIET=True)


    def uniprot_mapping(self, uniprot_path):
        '''
        Load a pdb-uniprot mapping in order to allow fast recognition and 
        exclusion of identical structures

        Input:
        path to a txt format uniprot dump (Swissprot is enough)

        Output:
        loads a self.uni_mapping dictionary with a set of uniprot ids for each pdb

        '''

        with open(uniprot_path) as uniprot:
            entries = ''.join([line for line in uniprot])

        uni_mapping = {}
        entries = entries.split('//')
        for entry in entries:

            entry = entry.split('\n')
            for line in entry:

                if line.startswith('AC   '):
                    uni_code = line[5:].strip(';').split(';')[0]

                elif line.startswith('DR   PDB;'):
                    pdb_code = line.split()[2].strip(';').lower()
                    if pdb_code not in uni_mapping: uni_mapping[pdb_code] = set()
                    uni_mapping[pdb_code].add(uni_code)

        self.uni_mapping = uni_mapping


    def parse_seqres(self, database_folder):
        '''
        Given a local sync of the pdb, find out complexes containing
        non-standard amino-acid chains (mod.res, DNA, RNA) and structures
        with less than 2 actual chain structures.

        Input:
        database_folder - path to the pdb biounits folder sync.
        Must have the following structure:
        
        database_folder/
            00/
                100a.pdb1
                100b.pdb1
                ...
            01/
                101a.pdb1
                101b.pdb1
                ...
            ...

        Output:
        seqres.pkl containing seqres mappings, chain number and filename 
        of selected structures
            
        faulty.pkl containing lists of omitted structures

        '''

######## compute path list
        path_list = []
        for pdb_code in self.pdblist:
            for n in range(1,10):
                pdb_path = f'{database_folder}/{pdb_code[1:3]}/{pdb_code}.pdb{n}'
                cif_path = f'{database_folder}/{pdb_code[1:3]}/{pdb_code}_assembly{n}.cif'
                if os.path.exists(pdb_path): path_list.append([pdb_code, pdb_path])
                elif os.path.exists(cif_path): path_list.append([pdb_code, cif_path])

        pdb_dict = {} 
        parsed_sets = []
        faults = {'N':[], 'U':[], 'M':[]}
######## cycle over paths
        for path_idx, (pdb_code, str_path) in enumerate(path_list):
            str_format = str_path.split('.')[-1]

############ compute progression
            actual = int((float(path_idx+1)/len(path_list))*100)
            progression = '# {}% done! '.format(actual)

############ check complex is in swissprot and the set of uniprot ids
############ it contains hasn't been parsed yet
            if pdb_code in self.uni_mapping: 
                if self.uni_mapping[pdb_code] in parsed_sets: 
                    print (progression, str_path, '### Discarded! Redundant')
                    continue

############ get seqres info from file header
            with open(str_path) as handle:
                if 'pdb' in str_format:
                    seqres = {record.id:str(record.seq)
                        for record in PdbSeqresIterator(handle)}
                if 'cif' in str_format:
                    seqres = {record.id:str(record.seq)
                        for record in CifSeqresIterator(handle)}

############ load structure
            if 'pdb' in str_format:
                struc = load_structure(str_path, self.pdbp)
            if 'cif' in str_format:
                struc = load_structure(str_path, self.cifp)
            if not struc: continue

############ parse structure chains
            str_chains = []
            for model in struc:
                for chain in model:
                    if is_chain_valid(chain):
                        str_chains.append(chain.get_id())

############ quality controls
            valid_assembly = True

############ check seqres is not containing DNA/RNA
            if valid_assembly:
                for idx, seq in seqres.items():
                    vocabulary = set([letter for letter in seq \
                        if letter in standard_residues or letter == 'U'])
                    if vocabulary == {'A', 'T', 'C', 'G'} \
                    or vocabulary == {'A', 'U', 'C', 'G'}:
                        valid_assembly = False
                        fault = 'Contains DNA/RNA!'
                        if pdb_code not in faults['N']:
                            faults['N'].append(pdb_code)
                        break

############ check PDB have no sequences with unknown residues
            if valid_assembly:
                unknown_residues = False
                for key, seq in seqres.items():
                    for residue in seq:
                        if residue not in standard_residues:
                            unknown_residues = True
                            break
                if unknown_residues:
                    valid_assembly = False
                    fault = 'Unknown residues!'
                    if pdb_code not in faults['U']:
                        faults['U'].append(pdb_code)


############ check seqres seqs have corresponding coordinates
############ and skip assemblies with only one matching seq/str
############ or containing inconsistencies
            if valid_assembly:
                seq_chains = list(seqres.keys())
                for chain in seq_chains:
                    if chain not in str_chains: del(seqres[chain])
                for chain in str_chains:
                    if chain not in seq_chains: valid_assembly = False

                if len(str_chains) <= 1:
                    valid_assembly = False
                    fault = 'Only 1 chain structure or str/seqres mismatch'
                    if pdb_code not in faults['M']:
                        faults['M'].append(pdb_code)

############ sum up quality controls
            if valid_assembly:
                print (progression, str_path)
            else:
                print (progression, str_path, '### Discarded!', fault)
                continue

############ store in object data
            pdb_dict[pdb_code] = {}
            pdb_dict[pdb_code]['chain_num'] = len(str_chains)
            pdb_dict[pdb_code]['seqres'] = copy.deepcopy(seqres)
            pdb_dict[pdb_code]['full_path'] = str_path
            if pdb_code in faults['N']: faults['N'].remove(pdb_code)
            if pdb_code in faults['U']: faults['U'].remove(pdb_code)
            if pdb_code in faults['M']: faults['M'].remove(pdb_code)
            if pdb_code in self.uni_mapping: 
                parsed_sets.append(self.uni_mapping[pdb_code])

######## write partial good structures data
            if len(pdb_dict)%1000:
                with open(self.data_folder+'/seqres.pkl', 'wb') as f:
                    pickle.dump(pdb_dict, f)

######## write good structures data
        with open(self.data_folder+'/seqres.pkl', 'wb') as f:
            pickle.dump(pdb_dict, f)

######## write faulty codes
        with open(self.data_folder+'/faulty.pkl', 'wb') as f:
            pickle.dump(faults, f)

######## summary
        print (f'Found:')
        print (f'{len(pdb_dict)} acceptable structures')
        print (f'{len(faults["N"])} structures containing nucleic acids')
        print (f'{len(faults["U"])} structures containing non-standard main sequence residues')
        print (f'{len(faults["M"])} structures with 1 chain only or str/seqres mismatch')


    def load_seqres(self):
        '''
        load seqres.pkl
        '''

######## check seqres.pkl exists
        assert os.path.exists(self.data_folder+'/seqres.pkl') \
            'parse_seqres() method has to be used first to generate seqres.pkl'

######## load seqres info
        with open(self.data_folder+'/seqres.pkl', 'rb') as f:
            seqres = pickle.load(f)
        return seqres


    def find_heteromers(self):
        '''
        go over all asseblies data contained in the seqres.pkl file and return a subset 
        of hetero assemblies pdb codes.

        '''
        
        heteromers = []
        seqres = self.load_seqres()
        for pdb in seqres:
############ if there is uniprot information, chech at least 2 different 
############ uniprot codes are present in the same pdb
            if pdb in self.uni_mapping:
                if len(self.uni_mapping[pdb]) > 1: heteromers.append(pdb)

############ otherwise run pairwise alignments all vs all using seqres
            else:
                het = False
                for idx1, c1 in enumerate(seqres[pdb]['seqres']):
                    for idx2, c2 in enumerate(seqres[pdb]['seqres']):
                        if idx1 <= idx2: continue
                        if seqres[pdb]['seqres'][c1] == seqres[pdb]['seqres'][c2]: continue
                        
                        seq1 = seqres[pdb]['seqres'][c1]
                        seq2 = seqres[pdb]['seqres'][c2]
                        alignments = pairwise2.align.globalds(seq1, seq2, blosum62, -10, -0.5)
                        aligned1 = alignments[0][0]
                        aligned2 = alignments[0][1] 
                        identity = sum([1 for pos1, pos2 in zip(aligned1, aligned2) \
                            if pos1 == pos2]) / len(aligned1)

######################## calculate alignment identity
                        if identity < 0.9: het = True
                        if het: break
                    if het: break
                if het: heteromers.append(pdb)
        return heteromers


    def create_str_folder(self, pdb_list, target_folder):
        '''
        save subset of pdb structures in a single folder

        '''

        seqres = self.load_seqres()
        for pdb in pdb_list:
            command = ['cp',
                       seqres[pdb]['full_path'],
                       f'{target_folder}/{pdb}.pdb']
            sp.run(command)


    def create_seq_folder(self, pdb_list, target_folder):
        '''
        fetch sequences of pdb chains and save them in a single folder

        '''
        
        seqres = self.load_seqres()
        for pdb in pdb_list:
            
            seq = ''
            str_path = seqres[pdb]['full_path']
            struc = load_structure(str_path, self.pdbp)
            for model in struc:
                for chain in model:
                    chain_id = chain.get_id()
                    for res in chain:
                        res_id = res.get_resname()
                        if res_id not in standard_residues_three: continue
                        seq += three_to_one[res_id]

                    with open(f'{target_folder}/{pdb}_{chain_id}.fasta') as out:
                        out.write(f'>{pdb}_{chain_id}\n{seq}')


    def mmseq_clustering(self, binary, seq_folder, target_folder):
        '''
        launch mmseq to cluster fasta seqs contained in a single folder

        '''
        
        command = [binary,
                   'createdb',
                   seq_folder,
                   target_folder+'/seqdb']
        sp.run(command)

        if not os.path.exists(self.tmp_folder+'/tmp'):
            os.mkdir(self.tmp_folder+'/tmp')

        command = [binary,
                   'cluster',
                   self.models_folder+'/seqdb',
                   self.models_fol<a$der+'/clustering',
                   self.tmp_folder+'/tmp',
                   '--cluster-reassign',
                   '1',
                   '--min-seq-id',
                   thr]
        sp.run(command)

        command = [binary,
                   'createtsv',
                   self.models_folder+'/seqdb',
                   self.models_folder+'/seqdb',
                   self.models_folder+'/clustering',
                   self.cluster_path]
        sp.run(command)

        for path in glob.glob(self.models_folder+'/clustering.*'):
            os.remove(path)
        shutil.rmtree(self.models_folder+'/tmp')



def main():

    DC = DataCleaner(sys.argv[1], sys.argv[2], /home/gabriele/)
    DC.uniprot_mapping(sys.argv[3])
    DC.parse_seqres(sys.argv[4])

#    sasa = ShrakeRupley()
#    cid = chains[0].get_id()
#    sasa.compute(struc, level='C')
#    print (struc[0][cid].sasa)        
#    for chain in chains:
#    sasa.compute(chain, level='C')
#    print (chain.sasa)
#    break
        

if __name__ == '__main__':

    main()

