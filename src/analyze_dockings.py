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

from Bio import pairwise2
from Bio.Align import substitution_matrices
blosum62 = substitution_matrices.load("BLOSUM62")

from itertools import combinations
#from tmtools import tm_align

import matplotlib.pyplot as plt
import seaborn as sb

class SubDockingQuality():
    def __init__(self, data_folder, tmp_folder):
        self.data_folder = data_folder
        self.tmp_folder = tmp_folder

    def get_groundtruth_chains(self, binary, structure, docking):
        str_dic = self.get_sequences(structure)
        dock_dic = self.get_sequences(docking)

        write_sequences(str_dic, self.tmp_folder+'/structure.fasta')
        write_sequences(dock_dic, self.tmp_folder+'/docking.fasta')

        strdb_path = format_mmseqdb(binary, self.tmp_folder+'/structure.fasta')
        dockdb_path = format_mmseqdb(binary, self.tmp_folder+'/docking.fasta')
        alignment_path = run_mmseq(binary, dockdb_path, strdb_path)

        with open(alignment_path) as align:
            for line in align: print (align)

    def get_sequences(self, struc):        
        seq_dic = {}
        for chain in unfold_entities(struc, 'C'):
            seq = [three_to_one(res.get_resname()) for res in chain]
            seq = ''.join(seq)
            seq_dic[chain.get_id()] = seq
        return seq_dic


    def write_sequences(self, seq_dic, out_path):
        with open(out_path) as out:
            for key, seq in seq_dic:
                out.write(f'>{key}\n{seq}\n')


    def format_mmseqdb(self, binary, input_path):
        output_path = input_path.split('.')[0]+'.db'
        command = [
            binary,
            'createdb',
            input_path,
            output_path]
        sp.run(command)
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
                '-c', cov]
        sp.run(command)

        command = [
            binary,
            'convertalis',
            dock_path,
            struc_path,
            self.tmp_folder+'/aln.db',
            self.tmp_folder+'/alignment.tsv',
            '--format-output',
            'query,target,pident,qaln,taln']
        sp.run(command)

        return self.tmp_folder+'/alignment.tsv'


    def map_structures(self, chain_dock, chain_true):
        print ('to implement!')

    def run_dockq(self):
        print ('to implement!')


    def plot_dockqs(self):
        print ('to implement!')


    def plot_plddts(self):
        print ('to implement!')



def main():
    self.cifp = MMCIFParser(QUIET=True)
    mmcifdb = '/proj/berzelius-2021-29/Database/pdb_mmcif/raw/'

    SDQ = SubDockingQuality(
        '/proj/berzelius-2021-29/users/x_gabpo/complex_assembly/data/innerhom_fasta/',
        '/proj/berzelius-2021-29/users/x_gabpo/tmp/')
    
    pdb_list = ['1b33']
    for pdb in pdb_list:
        structure_path = mmcifdb+pdb+'.cif'
        structure = cifp.get_structure('', structure_path)
        
        target_path = f'{self.data_folder}/{pdb}/{pdb}_*/ranked_0.pdb'
        for docking_path in glob.glob(target_path):
            docking = cifp.get_structure('', docking_path)

            SDQ.get_groundtruth_chains(
                '/proj/berzelius-2021-29/users/x_gabpo/mmseqs/bin/mmseqs',
                structure,
                docking)
        

if __name__ == '__main__':
    main()
