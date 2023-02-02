import os
import copy
import time
import glob
import json
import random
import argparse
import numpy as np
import pickle as pkl
import seaborn as sb
import matplotlib.pyplot as plt
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrn
from jax import vmap
from jax import tree_util
from replay_buffer import *
from environment import *
from features import *
from ops import *

from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Polypeptide import three_to_one
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Selection import unfold_entities

io = PDBIO()
pdbp = PDBParser(QUIET=True)
sasa = ShrakeRupley()

def set_idx_bvalue(idxs, struc):
    idx = 0
    for chain in struc[0]:
        for residue in chain:
            if 'CA' in residue:
                for atom in residue:
                    if idx in idxs: atom.set_bfactor(1)
                    else: atom.set_bfactor(0)
                idx += 1


def swap_chain(struc, new):
    chains = unfold_entities(struc, 'C')
    for chain in chains:
        idx = chain.get_id()
        struc[0].detach_child(idx)
        chain.detach_parent()
        new.add(chain)


def write_new_model(name, struc_rec, struc_lig):
    out_struc = Structure('out')
    out_model = Model(0)
    swap_chain(struc_rec, out_model)
    swap_chain(struc_lig, out_model)
    out_struc.add(out_model)
    io.set_structure(out_struc)
    io.save(name)


def out_ASA(code, struc_rec, struc_lig):
    key = jrn.PRNGKey(42)
    formatted_data = format_data(
            code, struc_rec, struc_lig, key, False)

    coord_rec = formatted_data[1]['coord'][0]
    mask_rec = formatted_data[1]['masks'][0]
    coord_lig = formatted_data[1]['coord'][1]
    mask_lig = formatted_data[1]['masks'][1]
    accessible = [idx for idx, asa in enumerate(mask_lig) if asa >= 0.2]

    print ('Set with ASA > 0.2:', accessible)

    idx = 0
    for chain in struc_rec[0]:
        for residue in chain:
            if 'CA' in residue:
                residue['CA'].set_coord(coord_rec[idx])
                for atom in residue:
                    if mask_rec[idx] >= 0.2: atom.set_bfactor(1)
                    else: atom.set_bfactor(mask_rec[idx])
                idx += 1

    idx = 0
    for chain in struc_lig[0]:
        for residue in chain:
            if 'CA' in residue:
                residue['CA'].set_coord(coord_lig[idx])
                for atom in residue:
                    if mask_lig[idx] >= 0.2: atom.set_bfactor(1)
                    else: atom.set_bfactor(mask_lig[idx])
                idx += 1

    write_new_model('test_ASA.pdb', struc_rec, struc_lig)


def out_interface(struc_rec, struc_lig, intmask_rec, intmask_lig):
    set_idx_bvalue(intmask_rec, struc_rec)
    set_idx_bvalue(intmask_lig, struc_lig)
    write_new_model('test_INT.pdb', struc_rec, struc_lig)


def out_rim(struc_rec, struc_lig, rimmask_rec, rimmask_lig):
    set_idx_bvalue(rimmask_rec, struc_rec)
    set_idx_bvalue(rimmask_lig, struc_lig)
    write_new_model('test_RIM.pdb', struc_rec, struc_lig)


def out_action(struc_rec, struc_lig, action):
    ligs = []
    for n, line in enumerate(action[0]):
        l, r1, r2, r3, d1, d2, d3 = line

        idxs = [r1, r2, r3]
        set_idx_bvalue(idxs, struc_rec)
        
        idxs = [l,]
        ligs.append(l)
        set_idx_bvalue(idxs, struc_lig)
       
        write_new_model(f'test_RL{n+1}.pdb', 
                copy.deepcopy(struc_rec), copy.deepcopy(struc_lig))

    set_idx_bvalue([], struc_rec)
    set_idx_bvalue(ligs, struc_lig)
    write_new_model('test_LIG.pdb', struc_rec, struc_lig)


def out_pose(name_id, struc_rec, struc_lig, R, t):
    atoms = jnp.array([atom.get_coord() for atom in unfold_entities(struc_lig, 'A')])
    atoms = (R@atoms.T).T
    atoms += t[None,:]

    idx = 0
    for chain in struc_lig[0]:
        for residue in chain:
            for atom in residue:
                atom.set_coord(atoms[idx])
                idx += 1

    write_new_model(f'test_ACT{name_id}.pdb', struc_rec, copy.deepcopy(struc_lig))
    return struc_lig

def is_rotation(M):
    return jnp.matmul(M, M.T), jnp.linalg.det(M)

if __name__ == "__main__":
    key = jrn.PRNGKey(42)
    data_path = '/home/pozzati/complex_assembly/data'
    test, test_code = load_dataset(data_path+'/dataset_features', size=1)
    print (test_code)

    path_rec = f'{data_path}/benchmark5.5/{test_code}_r_b.pdb'
    path_lig = f'{data_path}/benchmark5.5/{test_code}_l_b.pdb'
    
    struc_rec = pdbp.get_structure('', path_rec)
    struc_lig = pdbp.get_structure('', path_lig)

    print ('RES IDs of chain borders')
    for chain in unfold_entities(struc_rec, 'C')+unfold_entities(struc_lig, 'C'):
        print ('first', unfold_entities(chain, 'R')[0].get_id())
        print ('last', unfold_entities(chain, 'R')[-1].get_id())

    key = jrn.PRNGKey(42)
    env = DockingEnv(test, 10, 400, key)
    print ('Length of non-padding seqs')
    print (env.lengths_rec, env.lengths_lig)

    (edge_int, send_int, neigh_int,
     env.intmask_rec, env.intmask_lig, 
     env.rimmask_rec, env.rimmask_lig) = env.get_state(env.coord_lig)

    intmask_rec = jnp.squeeze(jnp.argwhere(env.intmask_rec==1)[:,1])
    intmask_lig = jnp.squeeze(jnp.argwhere(env.intmask_lig==1)[:,1])
    rimmask_rec = jnp.squeeze(jnp.argwhere(env.rimmask_rec==1)[:,1])
    rimmask_lig = jnp.squeeze(jnp.argwhere(env.rimmask_lig==1)[:,1])

    print ('rec int')
    print (intmask_rec)
    print ('lig int')
    print (intmask_lig)
    print ('rec rim')
    print (rimmask_rec)
    print ('lig rim')
    print (rimmask_lig)

    #action = env.get_random_action(key)
    #print ('action')
    #print (action)

    #new_coord_lig, R, t = env.step(env.coord_rec, env.coord_lig, action)

    ##############################################
    # test restoring coordinates and visualize ASA
    ##############################################

    #out_ASA(test_code, 
    #        copy.deepcopy(struc_rec), 
    #        copy.deepcopy(struc_lig))

    ############
    # interface
    ############

    #out_interface(
    #        copy.deepcopy(struc_rec), copy.deepcopy(struc_lig), 
    #        intmask_rec, intmask_lig)

    ######
    # rim
    ######

    #out_rim(copy.deepcopy(struc_rec), copy.deepcopy(struc_lig), 
    #        rimmask_rec, rimmask_lig)

    #######################################
    # visualize residues selected in action
    #######################################
    
    #out_action(copy.deepcopy(struc_rec), copy.deepcopy(struc_lig), action)

    #################
    # visualize step
    #################
    for n in range(10):
        action = env.get_random_action(key)
        env.coord_lig, R, t = env.step(env.coord_rec, env.coord_lig, action)
        I, det = is_rotation(R[0])
        print ('\n\n',I, det)

        struc_lig = out_pose(n, copy.deepcopy(struc_rec), struc_lig, R[0], t[0])
        
        (edge_int, send_int, neigh_int,
         env.intmask_rec, env.intmask_lig,
         env.rimmask_rec, env.rimmask_lig) = env.get_state(env.coord_lig)


