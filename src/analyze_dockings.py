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
from Bio.PDB.PDBIO import PDBIO
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


class DockingQualityManager():
    def __init__(self, data_folder, tmp_folder):
        self.data_folder = data_folder
        self.tmp_folder = tmp_folder
####################################

    def setup_chains(self, structure, docking):
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
            interfaces = self.get_chain_interfaces(self.structure, rec_chain)
            for lig_chain, rec_set in interfaces.items():
                if len(rec_set) == 0: continue
                if lig_chain in self.chain_map[chain2]:
                    rec_set = set([f'{res}R' for res in rec_set])
                    lig_set = self.get_interface(self.structure, lig_chain, rec_chain)
                    lig_set = set([f'{res}L' for res in lig_set])
                    relevant_int.append([rec_chain, lig_chain, rec_set.union(lig_set)])
        
        # sort interface by size 
        sizes = np.array([len(res_set) for c1, c2, res_set in relevant_int])
        relevant_int = [relevant_int[idx] for idx in np.argsort(sizes)[::-1]]
        
        # find largest interface and remove redundant ones
        nonredunant_int = []
        for c1, c2, set1 in relevant_int:
            redundant = False
            for c11, c22, set2 in nonredunant_int:
                common_res = set1.intersection(set2)
                if len(common_res) >= len(set1)*0.5: redundant = True
            #if not redundant: nonredunant_int.append([c1, c2, set1])
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
                out.write(f'>{key}\n{seq}\n')
############################################

    def get_chain_residues(self, structure, chain_id):
        for chain in unfold_entities(structure, 'C'):
            if chain.get_id() == chain_id:
                return unfold_entities(chain, 'R')
##################################################

    def get_chain_interfaces(self, structure, chain_id):
        interface_dic = {}
        for chain in unfold_entities(structure, 'C'):
            if chain.get_id() == chain_id: continue
            interface = self.get_interface(structure, chain_id, chain.get_id())
            interface_dic[chain.get_id()] = interface
        return interface_dic
############################

    def get_interface(self, structure, rec_chain_id, lig_chain_id): 
        lig_residues = self.get_chain_residues(structure, lig_chain_id)
        lig_atom_list = [atom for residue in lig_residues for atom in residue \
                         if atom.get_id() == 'CA']
        ns = NeighborSearch(lig_atom_list)
        
        interface = set()
        rec_residues = self.get_chain_residues(structure, rec_chain_id)
        for residue in rec_residues:
            if residue.get_id()[0] != ' ': continue 
            for atom in unfold_entities(residue, 'A'):
                if atom.get_id() != 'CA': continue
                n = ns.search(atom.get_coord(), 10, level='R')
                if len(n) != 0: interface.add(residue.get_id()[1])

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

    def seq_identity(self, alignment):
        acc = 0.0
        for res1, res2 in zip(alignment[0], alignment[1]):
            if res1==res2: acc += 1
        return acc/len(alignment[0])

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
                    elif res1 == '-':
                        struc_res[idx].detach_parent()
                        continue
                    idx += 1
############################

    def get_pdockq(self, struc):
        rec, lig = unfold_entities(struc, 'C')
        rec_res = self.get_interface(struc, rec.get_id(), lig.get_id())
        lig_res = self.get_interface(struc, lig.get_id(), rec.get_id())

        plddts = [rec[idx]['CA'].get_bfactor() for idx in rec_res]
        plddts += [lig[idx]['CA'].get_bfactor() for idx in lig_res]
        IF_plddt = sum(plddts)/len(plddts)
        x = np.log(len(plddts)+1.e-20)*IF_plddt

        L = 7.07140240e-01
        x0 = 3.88062162e+02
        k = 3.14767156e-02
        b = 3.13182907e-02

        pDockQ = L / (1 + np.exp(-k*(x-x0)))+b

        return pDockQ, IF_plddt, len(plddts)
############################################

    def get_dockq(self, dockq_path, target_path):
        out_manager = PDBIO()
        # get native interaction chains
        for idx1, drec_chain_id in enumerate(self.chain_map.keys()):
            for idx2, dlig_chain_id in enumerate(self.chain_map.keys()):
                if idx1 >= idx2: continue
                nonredundant_int = self.derive_ground_truth(drec_chain_id, dlig_chain_id)

                if len(nonredundant_int) == 0: return None, [None, None]

                max_dockq = 0
                for nrec_chain_id, nlig_chain_id, size in nonredundant_int:
                    # write down native in TMP matching docking model chain id
                    native = Structure('native')
                    native_model = Model(0)
                    native.add(native_model)
                    structure_decoy = copy.deepcopy(self.structure)
                    for chain in unfold_entities(structure_decoy, 'C'):
                        if chain.get_id() == nrec_chain_id:
                            chain.detach_parent()
                            chain.id = drec_chain_id
                            native[0].add(chain)
                        elif chain.get_id() == nlig_chain_id:
                            chain.detach_parent()
                            chain.id = dlig_chain_id
                            native[0].add(chain)
                    
                    out_manager.set_structure(native)
                    out_manager.save(self.tmp_folder+'native.pdb')

                    # run dockq
                    command = ['python3',
                               dockq_path,
                               '-short',
                               target_path,
                               self.tmp_folder+'native.pdb']
                    result = sp.run(command, capture_output=True, text=True)
                    dockq = float(result.stdout.split()[1])
                    if dockq >= max_dockq: 
                        max_dockq = dockq
                        chain_match = [nrec_chain_id, nlig_chain_id]
                #        command = [mv, 
                #                   self.tmp_folder+'native.pdb',
                #                   self.tmp_folder+'best_native.pdb']
                #        sp.run(command, stdout=sp.DEVNULL)

                #n = dockq_path[-5]
                #docking_folder = '/'.join(dockq_path.split('/')[:-1])

                #command = [mv,
                #           self.tmp_folder+'best_native.pdb',
                #           '/'.join(dockq_path.split()[:-1]),]
                #sp.run(command, stdout=sp.DEVNULL)
                return max_dockq, chain_match


def main():
    pdbp = PDBParser(QUIET=True)
    cifp = MMCIFParser(QUIET=True)
    mmcifdb = '/proj/berzelius-2021-29/Database/pdb_mmcif/raw/'
    mmseqs_path = '/proj/berzelius-2021-29/users/x_gabpo/programs/mmseqs/bin/mmseqs'
    dockq_path = '/proj/berzelius-2021-29/users/x_gabpo/programs/DockQ/DockQ.py'

    pdb = sys.argv[1]

    DQM = DockingQualityManager(
        '/proj/berzelius-2021-29/users/x_gabpo/complex_assembly/data/innerhom_fasta/',
        f'/proj/berzelius-2021-29/users/x_gabpo/tmp/tmp_{pdb}/')
    
    output_path = f'/proj/berzelius-2021-29/users/x_gabpo/complex_assembly/data/innerhom_fasta/{pdb}/{pdb}_results.csv'
    with open(output_path, 'w') as outfile:
        outfile.write(f'PDB,Chain1,Chain2,Model,seqID,IFsize,IFplddt,dockq,pdockq\n')

    structure_path = mmcifdb+pdb+'.cif'
    structure = cifp.get_structure('', structure_path)
    
    pdb = sys.argv[1]
    target_path = f'{DQM.data_folder}/{pdb}/{pdb}_*/'
    for docking_folder in glob.glob(target_path):
        for n in range(5):
                
            docking_path = f'{docking_folder}/ranked_{n}.pdb'
            if not os.path.exists(docking_path): continue
            docking = pdbp.get_structure('', docking_path)
            
            DQM.setup_chains(structure, docking)
            DQM.map_docking_to_structure(mmseqs_path)
            
            dockq, [chain1, chain2] = DQM.get_dockq(dockq_path, docking_path)
                
            if chain1 is not None:
                seq1 = DQM.str_dic[chain1]
                seq2 = DQM.str_dic[chain2]
                seq_id = DQM.seq_identity(DQM.align_sequences(seq1, seq2))
                seq_id = round(seq_id, 3)
            else: seq_id = None

            pdockq, if_plddt, if_size = DQM.get_pdockq(DQM.docking)
                
            with open(output_path, 'a') as outfile:
                outfile.write(f'{pdb},{chain1},{chain2},ranked_{n},'\
                              f'{seq_id},{if_size},'\
                              f'{round(if_plddt,3)},{dockq},'\
                              f'{round(pdockq,3)}\n')
        

if __name__ == '__main__':
    main()
