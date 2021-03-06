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
import shutil
import numpy as np
import subprocess as sp
import multiprocessing as mp

from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure
from Bio.PDB.Selection import unfold_entities
from Bio.PDB.SASA import ShrakeRupley

from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.SeqIO.PdbIO import PdbSeqresIterator
from Bio.SeqIO.PdbIO import CifSeqresIterator

from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Polypeptide import three_to_one

from Bio import pairwise2
from Bio.Align import substitution_matrices
blosum62 = substitution_matrices.load("BLOSUM62")

from itertools import combinations
#from tmtools import tm_align

import matplotlib.pyplot as plt
import seaborn as sb

standard_residues = [
    'A','R','N','D','C','Q','E','G','H','I',
    'L','K','M','F','P','S','T','W','Y','V']

standard_residues_three = [
    'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']

def is_chain_valid(chain):

    # check there is some aminoacid
    polypep = False
    for residue in chain:
        if is_aa(residue):
            polypep = True
            break

    # check there is no weird stuff with ATOM record
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



def progression(label=''):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for progression in func(*args, **kwargs):
                print (f'{label} progression: {progression}')
        return wrapper
    return decorator



@timeout(60)
def load_structure(path, parser):
    return parser.get_structure('', path)



class DatasetManager():

    def __init__(self, 
                 list_path, 
                 data_folder,
                 database_folder,
                 tmp_folder):

        with open(list_path) as list_file:
            self.pdb_list = [line.strip() for line in list_file]

        self.tmp_folder = tmp_folder
        self.data_folder = data_folder
        self.database_folder = database_folder
        if not os.path.exists(tmp_folder): os.mkdir(tmp_folder)
        self.cifp = MMCIFParser(QUIET=True)
        self.sasa = ShrakeRupley()

    def __call__(self,
                 min_chain=2,
                 max_chain=None,
                 innerhom=False,
                 homomer=True,
                 heteromer=True,
                 min_innerhom=None,
                 excl_nucl=True
                 excl_unknown_residues=True):

        # check already computed stuff
        if os.path.exists(self.data_folder+'/seqres.pkl'):
            with open(self.data_folder+'/seqres.pkl', 'rb') as pkl_file:
                pdb_dict = pkl.load(pkl_file)
        else: pdb_dict = {'parsed':[]}

        #compute path list
        results = []
        job_list = []
        for pdb_code in self.pdblist:
            if pdb_code in pdb_dict['parsed']: continue
            cif_path = f'{self.data_folder}/{pdb_code}.cif'
            job = [pdb_code, cif_path, excl_nucl, excl_unknown_residues]
            if os.path.exists(cif_path): job_list.append([job])
            if len(job_list) % 1000 == 0:
                p = mp.pool(25)
                results.extend(p.map(self.cleaning_worker, job_list))
                p.join()
                job_list = []



#        #compute progression
#            actual = int((float(path_idx+1)/len(path_list))*100)
#            progression = '# {}% done! '.format(actual)

            #store in object data
            pdb_dict[pdb_code] = {}
            pdb_dict[pdb_code]['seqres'] = copy.deepcopy(seqres)
            pdb_dict['parsed'].append(pdb_code)
            if pdb_code in self.uni_mapping:
                parsed_sets.append(self.uni_mapping[pdb_code])

            #write partial good structures data
            if len(pdb_dict)%1000:
                with open(self.data_folder+'/seqres.pkl', 'wb') as f:
                    pickle.dump(pdb_dict, f)

        #write good structures data
        with open(self.data_folder+'/seqres.pkl', 'wb') as f:
            pickle.dump(pdb_dict, f)



    def cleaning_worker(self, job):
        pdb_code = job[0]
        cif_path = job[1]
        excl_nucl = job[2]
        excl_unknown_residues = job[3]

        #check complex is in swissprot and the set of uniprot ids
        #it contains hasn't been parsed yet
        if pdb_code in self.uni_mapping: 
            if self.uni_mapping[pdb_code] in parsed_sets: 
                return [False, pdb_code, 'Redundant!']

        #get seqres info from file header and structure
        with open(str_path) as handle:
            seqres = {record.id:str(record.seq)
                for record in CifSeqresIterator(handle)}

        struc = load_structure(str_path, self.cifp)
        if not struc:
            return [False, pdb_code, 'No structure found!']

        #parse structure chains
        str_chains = []
        for model in struc:
            for chain in model:
                if is_chain_valid(chain):
                    str_chains.append(chain.get_id())

        #quality controls
        valid_assembly = True

        #check seqres is not containing DNA/RNA
        if valid_assembly and excl_nucl:
            for idx, seq in seqres.items():
                vocabulary = set([letter for letter in seq \
                    if letter in standard_residues or letter == 'U'])
                if vocabulary == {'A', 'T', 'C', 'G'} \
                or vocabulary == {'A', 'U', 'C', 'G'}:
                    valid_assembly = False
                    fault = 'Contains DNA/RNA!'

        #check PDB have no sequences with unknown residues
        if valid_assembly and excl_unknown_residues:
            unknown_residues = False
            for key, seq in seqres.items():
                for residue in seq:
                    if residue not in standard_residues:
                        unknown_residues = True
                        break
            if unknown_residues:
                valid_assembly = False
                fault = 'Unknown residues!'

        #check seqres seqs have corresponding coordinates
        #and skip assemblies with only one matching seq/str
        #or containing inconsistencies
        if valid_assembly:
            seq_chains = list(seqres.keys())
            for chain in seq_chains:
                if chain not in str_chains: del(seqres[chain])
            if len(str_chains) <= 1:
                valid_assembly = False
                fault = 'Only 1 chain structure or str/seqres mismatch'

        #sum up quality controls
        if not valid_assembly: return [False, pdb_code, fault]

        #compute cmap
        chains = unfold_entities(struc, 'R')
        for idx1, chain1 in enumerate(chains):
            for idx2, chain2 in enumerate(chains):
                if idx1 <= idx2: continue
                sasa.compute(chain1, level='R')
                sasa.compute(chain2, level='R')


        #store in object data and return
        result = {}
        result['seqres'] = copy.deepcopy(seqres)
        result['cmaps'] = 

        return [True, pdb_code, result]


    def load_seqres(self):
        '''
        load seqres.pkl

        '''

        #check seqres.pkl exists
        assert os.path.exists(self.data_folder+'/seqres.pkl'), \
            '__call__ method has to be used first to generate seqres.pkl'

        #load seqres info
        with open(self.data_folder+'/seqres.pkl', 'rb') as f:
            seqres = pickle.load(f)
        return seqres


    def group_fasta(self, 
                    pdb_list,
                    output_path,
                    af_foldtree=False):
        '''
        group selection of fasta sequences in a single or multiple fasta files

        '''

        if output_path.endswith('.fasta') or af_format:
            all_fasta = ''
            one_per_fasta = False
        else: 
            if not os.path.exists(output_path): os.mkdir(output_path)
            one_per_fasta = True

        seqres = self.load_seqres()
        for pdb in pdb_list:
            for chain_id, seq in seqres[pdb]['seqres'].items():
                if one_per_fasta:
                    with open(f'{output_path}/{pdb}_{chain_id}.fasta', 'w') as out:
                        out.write(f'>{pdb}_{chain_id}\n{seq}')
                else:
                    all_fasta += f'>{pdb}_{chain_id}\n{seq}\n'
            
            if af_format:
                pdb_output_path = f'{output_path}/{pdb}.fasta'
                with open(pdb_output_path, 'w') as out: out.write(all_fasta)
                all_fasta = ''

        if not one_per_fasta and not af_format:
            with open(output_path, 'w') as out: out.write(all_fasta) 


    def uniprot_mapping(self,
                        codes_list):

        url = 'https://www.uniprot.org/uploadlists/'

        params = {
            'from': 'PDB_ID',
            'to': 'ACC',
            'format': 'tab',
            'query': codes_list}

        data = urllib.parse.urlencode(params)
        data = data.encode('utf-8')
        req = urllib.request.Request(url, data)
        with urllib.request.urlopen(req) as f:
            response = f.read()
        print(response.decode('utf-8'))


    def chain_similarity(self,
                         binary,
                         input_path):
        '''
        run mmseq2/foldseek comparing all vs all chains in a complex/set of complexes

        '''

        assert 'mmseqs' in binary or 'foldseek' in binary, \
            'binary path needs to contain "mmseq" or "foldseek"'

        command = [
            binary,
            'createdb',
            input_path,
            self.tmp_folder+'/seqdb']
        sp.run(command)

        if 'mmseqs' in binary: self.run_mmseq(binary)
        elif 'foldseek' in binary: self.run_foldseek(binary)

        for path in glob.glob(self.tmp_folder+'/*'):
            try: os.remove(path)
            except: shutil.rmtree(path)

        scores = {}
        if 'mmseqs' in binary: result_path = self.data_folder+'identities.tsv'
        elif 'foldseek' in binary: result_path = self.data_folder+'tmscores.tsv'
        for line in open(result_path):
            line = line.strip()
            if line == '': continue

            pdb1 = line.split()[0][:4]
            c1 = line.split()[0].split('_')[1]
            
            pdb2 = line.split()[1][:4]
            c2 = line.split()[1].split('_')[1]
            score = float(line.split()[2])
            
            scores[pdb1] = scores.get(pdb1, {})
            scores[pdb1][c1] = scores[pdb1].get(c1, {})
            scores[pdb1][c1][pdb2] = scores[pdb1][c1].get(pdb2, {})
            scores[pdb1][c1][pdb2][c2] = score

        return scores



def main():
    mmseq_bin = '/proj/nobackup/snic2019-35-62/gpozzati/programs/mmseqs'
    foldseek_bin = '/proj/nobackup/snic2019-35-62/gpozzati/programs/foldseek'
    database_path = '/proj/nobackup/snic2019-35-62/Database/pdb_mmcif/pdb_mmcif'
    
    DC = DataCleaner('/proj/nobackup/snic2019-35-62/gpozzati/complex_assembly/data/multimers',
                     '/proj/nobackup/snic2019-35-62/gpozzati/complex_assembly/data/',
                     '/proj/nobackup/snic2019-35-62/gpozzati/tmp/')
    
    seqres = DC.load_seqres()
    pdb_list = list(seqres.keys())
    DC.group_fasta(pdb_list, DC.tmp_folder+'/all.fasta')

    identities = DC.chain_similarity(mmseq_bin, DC.tmp_folder+'/all.fasta')

    multimeric_state = {}
    for pdb in pdb_list:
        stop = False
        for chain1 in seqres[pdb]['seqres']:
            for chain2 in identities[pdb][chain1][pdb]:
                if identities[pdb][chain1][pdb][chain2] < 90:
                    multimeric_state[pdb] = 'het'
                    stop = True
                    break
            if stop: break
        if not stop: multimeric_state[pdb] = 'hom'

    het_folder = DC.data_folder+'/heteromers/' 
    if not os.path.exists(het_folder): os.mkdir(het_folder)

    het_list = []
    for pdb in pdb_list:
        if multimeric_state[pdb] == 'hom': continue
        pdb_path = f'{database_path}/{pdb}.cif'
        if not os.path.exists(pdb_path): 
            print (pdb_path,'not found!')
            continue
        shutil.copy(pdb_path, het_folder)
        het_list.append(pdb)

    with open(DC.data_folder+'/hetero_list','w') as out:
        for pdb in het_list: out.write(pdb+'\n')

    chain_dist = []
    innerhom_list = []
    tmscores = DC.chain_similarity(foldseek_bin, het_folder)
    for pdb in het_list:
        if pdb not in tmscores: continue
        for chain1 in tmscores[pdb]:
            for chain2 in tmscores[pdb][chain1][pdb]:
                if tmscores[pdb][chain1][pdb][chain2] > 0.6:
                    innerhom_list.append(pdb)
                    chain_dist.append(seqres[pdb]['chain_num'])
                    stop = True
                    break
            if stop: break
    
    with open(DC.data_folder+'/innerhom_list','w') as out:
        for pdb in innerhom_list: out.write(pdb+'\n')

    innerhom_folder = DC.data_folder+'/innerhom_fasta/'
    if not os.path.exists(innerhom_folder): os.mkdir(innerhom_folder)
    DC.group_fasta(innerhom_list, innerhom_folder, af_format=True)

    sb.distplot(chain_dist, kde=False, bins=range(0,101))
    plt.xlim(0,100)

    plt.savefig('chain_number_dist.png')


if __name__ == '__main__':

    main()

