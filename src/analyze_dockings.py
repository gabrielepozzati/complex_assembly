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

from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure
from Bio.PDB.Selection import unfold_entities

from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.SeqIO.PdbIO import PdbSeqresIterator
from Bio.SeqIO.PdbIO import CifSeqresIterator

#from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Polypeptide import three_to_one

from Bio.pairwise2 import align
from Bio.Align import substitution_matrices
blosum62 = substitution_matrices.load("BLOSUM62")

from itertools import combinations
#from tmtools import tm_align

standard_three = ['ALA', 'ARG', 'ASN', 'ASP', 
                  'CYS', 'GLN', 'GLU', 'GLY', 
                  'HIS', 'ILE', 'LEU', 'LYS', 
                  'MET', 'PHE', 'PRO', 'SER', 
                  'THR', 'TRP', 'TYR', 'VAL']


class SubDockingQuality():
    def __init__(self, data_folder, tmp_folder):
        self.data_folder = data_folder
        self.tmp_folder = tmp_folder


    def setup_chains(self, binary, structure, docking):
        self.docking = docking
        self.structure = structure

        # get sequences from the actual structures 
        # in the docking and original complexes
        self.str_dic = self.get_sequences(self.structure)
        self.dock_dic = self.get_sequences(self.docking)

        # run mmseq to map all docking chains vs all true structure chains
        self.write_sequences(self.str_dic, self.tmp_folder+'/structure.fasta')
        self.write_sequences(self.dock_dic, self.tmp_folder+'/docking.fasta')
        strdb_path = self.format_mmseqdb(binary, self.tmp_folder+'/structure.fasta')
        dockdb_path = self.format_mmseqdb(binary, self.tmp_folder+'/docking.fasta')
        alignment_path = self.run_mmseq(binary, dockdb_path, strdb_path)

        # parse mmseq output
        chain_map = {}
        with open(alignment_path) as align:
            best_identity = 0
            for line in align:
                ch_dock, ch_str, identity = line.split('\t')
                identity = float(identity)
                
                if ch_dock not in chain_map: 
                    best_identity = 0
                    str_chains = []
                if identity > best_identity: 
                    str_chains = []
                
                if identity >= best_identity:
                    best_identity = identity
                    str_chains.append(ch_str)
                    chain_map[ch_dock] = list(str_chains)
        
        # modify residue numbering in structure to match docking
        self.map_structures(chain_map)

        for path in glob.glob(self.tmp_folder+'/*'):
            try: os.remove(path)
            except: shutil.rmtree(path)


    def get_sequences(self, struc):
        seq_dic = {}
        for chain in unfold_entities(struc, 'C'):
            seq = ''
            for res in chain:
                if res.get_id()[0] != ' ': continue
                if res.get_resname() in standard_three: 
                    seq += three_to_one(res.get_resname())
                else: seq += 'X'

            seq_dic[chain.get_id()] = seq
        return seq_dic


    def write_sequences(self, seq_dic, out_path):
        with open(out_path, 'w') as out:
            for key, seq in seq_dic.items():
                out.write(f'>{key}\n{seq}\n')


    def format_mmseqdb(self, binary, input_path):
        output_path = input_path.split('.')[0]+'.db'
        command = [
            binary,
            'createdb',
            input_path,
            output_path]
        sp.run(command, stdout=sp.DEVNULL)

        return output_path


    def run_mmseq(self,
                  binary,
                  dock_path,
                  struc_path,
                  cov='0.0',
                  seqid='0.0'):

        command = [
                binary,
                'map',
                dock_path,
                struc_path,
                self.tmp_folder+'/aln.db',
                self.tmp_folder, '-a',
                '--min-seq-id', seqid,
                '--rescore-mode', '3',
                '-c', cov]
        sp.run(command, stdout=sp.DEVNULL)

        command = [
            binary,
            'convertalis',
            dock_path,
            struc_path,
            self.tmp_folder+'/aln.db',
            self.tmp_folder+'/alignment.tsv',
            '--format-output',
            'query,target,pident']
        sp.run(command, stdout=sp.DEVNULL)

        return self.tmp_folder+'/alignment.tsv'


    def align_sequences(self, sequence1, sequence2):
        alignment = align.globalds(sequence1, sequence2, blosum62, -11, -1)
        return alignment[0]


    def get_residues(self, structure, chain_id):
        for chain in unfold_entities(structure, 'C'):
            if chain.get_id() == chain_id: 
                return unfold_entities(chain, 'R')


    def print_residue_id(self, structure, chain_id):
        for model in structure:
            for chain in model:
                if chain.id == chain_id:
                    for residue in chain:
                        if residue.get_id()[0] == ' ':
                            print(residue.id)

    def map_structures(self, chain_map):
        for dock_id, chains in chain_map.items():
            dock_res = self.get_residues(self.docking, dock_id)
            for struc_id in chains:
                struc_res = self.get_residues(self.structure, struc_id)
                
                s1 = self.dock_dic[dock_id]
                s2 = self.str_dic[struc_id]
                align = self.align_sequences(s1, s2)
                
                idx = 0
                for res1, res2 in zip(align[0], align[1]):
                    if res1 != '-' and res2 != '-':
                        struc_res[idx].id = dock_res[idx].id
                    elif r1 == '-':
                        struc_res[idx].detach_parent()
                        continue
                    idx += 1

    def run_dockq(self):
        print ('to implement!')


    def plot_dockqs(self):
        print ('to implement!')


    def plot_plddts(self):
        print ('to implement!')



def main():
    pdbp = PDBParser(QUIET=True)
    cifp = MMCIFParser(QUIET=True)
    mmcifdb = '/proj/berzelius-2021-29/Database/pdb_mmcif/raw/'

    SDQ = SubDockingQuality(
        '/proj/berzelius-2021-29/users/x_gabpo/complex_assembly/data/innerhom_fasta/',
        '/proj/berzelius-2021-29/users/x_gabpo/tmp/')
    
    pdb_list = ['1b33']
    for pdb in pdb_list:
        structure_path = mmcifdb+pdb+'.cif'
        structure = cifp.get_structure('', structure_path)
        
        target_path = f'{SDQ.data_folder}/{pdb}/{pdb}_*/ranked_0.pdb'
        for docking_path in glob.glob(target_path):
            docking = pdbp.get_structure('', docking_path)
            
            SDQ.setup_chains(
                '/proj/berzelius-2021-29/users/x_gabpo/mmseqs/bin/mmseqs',
                structure,
                docking)
            break
        

if __name__ == '__main__':
    main()