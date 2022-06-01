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
from Bio.PDB.NeighborSearch import NeighborSearch
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


class AFDockingQuality():
    def __init__(self, data_folder, tmp_folder):
        self.data_folder = data_folder
        self.tmp_folder = tmp_folder
####################################

    def setup_chains(self, binary, structure, docking):
        self.docking = docking
        self.structure = structure

        # get sequences from the actual structures 
        # in the docking and original complexes
        self.str_dic = self.get_sequences(self.structure)
        self.dock_dic = self.get_sequences(self.docking)
########################################################

    def map_docking_to_structure(self, binary):
        # run mmseq to map all docking chains vs all true structure chains
        self.write_sequences(self.str_dic, self.tmp_folder+'/structure.fasta')
        self.write_sequences(self.dock_dic, self.tmp_folder+'/docking.fasta')
        strdb_path = self.format_mmseqdb(binary, self.tmp_folder+'/structure.fasta')
        dockdb_path = self.format_mmseqdb(binary, self.tmp_folder+'/docking.fasta')
        alignment_path = self.run_mmseq(binary, dockdb_path, strdb_path)

        # parse mmseq output
        chain_map = {}
        all_identities = []
        with open(alignment_path) as align:
            best_identity = 0
            for line in align:
                ch_dock, ch_str, identity = line.split('\t')
                identity = float(identity)
                all_identities.append([ch_dock, ch_str, identity])

                if ch_dock not in chain_map: 
                    best_identity = 0
                    str_chains = []
                if identity > best_identity: 
                    str_chains = []
                
                if identity >= best_identity:
                    best_identity = identity
                    str_chains.append(ch_str)
                    chain_map[ch_dock] = list(str_chains)
        
        self.chain_map = chain_map
        self.all_identities = all_identities

        # modify residue numbering in structure to match docking
        self.map_structures()

        # clean up tmp directory
        for path in glob.glob(self.tmp_folder+'/*'):
            try: os.remove(path)
            except: shutil.rmtree(path)
#######################################

    def derive_ground_truth(self, chain1, chain2):
        # find interfaces
        relevant_int = []
        for rec_chain in self.chain_map[chain1]:
            interfaces = get_interfaces(self.structure, rec_chain)
            for lig_chain, rec_set in interfaces.items():
                if lig_chain in self.chain_map[chain2]:
                    rec_set = set([f'{res}R' for res in rec_set])
                    lig_set = self.get_interface(structure, lig_chain, rec_chain)
                    lig_set = set([f'{res}L' for res in lig_set])
                    relevant_int.append([rec_chain, lig_chain, rec_set.union(lig_set)])
        
        # sort interface by size 
        sizes = [len(res_set) for c1, c2, res_set in relevant_int]
        sizes = np.array(sizes)
        relevant_int = relevant_int[np.argsort(sizes)[::-1]]
        
        # find largest interface and remove redundant ones
        nonredunant_int = []
        for c1, c2, set1 in relevant_int:
            for c11, c22, set2 in nonredunant_int:
                common_res = set1.intersection(set2)
                if len(common_res) >= len(set1)*0.5: continue
            nonredunant_int.append([c1, c2, set1])
        return nonredunant_int
##############################

    def get_sequences(self, structure):
        seq_dic = {}
        for chain in unfold_entities(structure, 'C'):
            seq = ''
            for res in chain:
                if res.get_id()[0] != ' ': continue
                if res.get_resname() in standard_three:
                    seq += three_to_one(res.get_resname())
                else: seq += 'X'

            seq_dic[chain.get_id()] = seq
        return seq_dic
######################

    def write_sequences(self, seq_dic, out_path):
        with open(out_path, 'w') as out:
            for key, seq in seq_dic.items():
                out.write(f'>{key}\n{seq}\n'
############################################

    def get_chain_residues(self, structure, chain_id):
        for chain in unfold_entities(structure, 'C'):
            if chain.get_id() == chain_id:
                return unfold_entities(chain, 'R')
##################################################

    def get_chain_interfaces(self, structure, chain_id):
        interface_dic = {}
        for chain in unfold_entities(structure, 'C')
            if chain.get_id() == chain_id: continue
            interface = self.get_interface(structure, chain_id, chain.get_id())
            interface_dic[chain.get_id()] = interface
        return interface_dic
############################

    def get_interface(self, structure, rec_chain_id, lig_chain_id): 
        lig_atom_list += [list(atom.get_coord()) \
            for atom in unfold_entities(lig_chain, 'A')]
        ns = NeighborSearch(ligand_atom_list)
        
        interface = set()
        rec_residues = get_chain_residues(structure, rec_chain)
        for residue in rec_residues:
            for atom in unfold_entities(residue, 'A'):
                n = ns.search(atom.get_coords(), 6, level='R')
                if len(n) != 0:
                    interface.add(residue.get_id()[1])
                    break
        return interface
########################

    def format_mmseqdb(self, binary, input_path):
        output_path = input_path.split('.')[0]+'.db'
        command = [
            binary,
            'createdb',
            input_path,
            output_path]
        sp.run(command, stdout=sp.DEVNULL)

        return output_path
##########################

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
###############################################

    def align_sequences(self, sequence1, sequence2):
        alignment = align.globalds(sequence1, sequence2, blosum62, -11, -1)
        return alignment[0]
###########################

    def superpose(structure1, structure2):
        superpose_structure = Bio.PDB.Superimposer()
        superpose_structure.set_atoms(structure1, structure2)
        superpose_structure.apply(structure1)
        return superpose_structure
##############################

    def map_structures(self):
        for dock_id, chains in self.chain_map.items():
            dock_res = self.get_chain_residues(self.docking, dock_id)
            for struc_id in chains:
                struc_res = self.get_chain_residues(self.structure, struc_id)
                
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
############################

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
