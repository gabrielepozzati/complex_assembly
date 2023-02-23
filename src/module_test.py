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
from functools import partial
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


import jax
import jax.numpy as jnp
import jax.random as jrn
from jax import vmap
from jax import tree_util
from replay_buffer import *
from quaternions import *
from environment import *
from features import *
from ops import *

from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.Atom import Atom
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


def generate_random_sequence(key, sruc_lig, env, N=500):

    new_model = copy.deepcopy(struc_lig[0])
    new_model.detach_parent()

    count = 0
    key, keyn = jrn.split(key, 2)
    actions = env.get_random_action(keyn, env.intmask_ligs,
            env.rimmask_recs, env.rimmask_ligs)
    Ps, p0s, p1s = env.refine_action(env.coord_ligs, actions)
    quats = vmap(quat_from_pivoting)(Ps, p0s, p1s)

    for n in range(N):
        print (f'####################### ACTION {n}')
        
        intmask_recs = jnp.squeeze(jnp.argwhere(env.intmask_recs==1)[:,1])
        intmask_ligs = jnp.squeeze(jnp.argwhere(env.intmask_ligs==1)[:,1])
        rimmask_recs = jnp.squeeze(jnp.argwhere(env.rimmask_recs==1)[:,1])
        rimmask_ligs = jnp.squeeze(jnp.argwhere(env.rimmask_ligs==1)[:,1])
        print ('rec int', intmask_recs)
        print ('lig int', intmask_ligs)
        print ('rec rim', rimmask_recs)
        print ('lig rim', rimmask_ligs)

        print ('CHECK ACTION...')
        coord_ligs = env.step(env.coord_ligs, Ps, quats)

        dmaps = vmap(distances_from_coords)(
                env.coord_recs[:,:,None,:], coord_ligs[:,None,:,:])
        dmaps = vmap(lambda x,y,z: x*y[:,None]*z[None,:])(
                dmaps, env.padmask_recs, env.padmask_ligs)
        dmaps = jnp.where(dmaps!=0, dmaps, 1e6)

        notice = True
        while jnp.min(dmaps)<3 or jnp.min(dmaps)>8 or count==24:
            if notice: print ('SCREENING...')
            key, keyn = jrn.split(key, 2)
            actions = env.get_random_action(keyn, env.intmask_ligs,
                    env.rimmask_recs, env.rimmask_ligs)
            Ps, p0s, p1s = env.refine_action(env.coord_ligs, actions)
            quats = vmap(quat_from_pivoting)(Ps, p0s, p1s)

            coord_ligs = env.step(env.coord_ligs, Ps, quats)

            dmaps = vmap(distances_from_coords)(
                    env.coord_recs[:,:,None,:], coord_ligs[:,None,:,:])
            dmaps = vmap(lambda x,y,z: x*y[:,None]*z[None,:])(
                    dmaps, env.padmask_recs, env.padmask_ligs)
            dmaps = jnp.where(dmaps!=0, dmaps, 1e6)
            notice = False
            count = 0
        print ('OK! --- Action ->', actions)

        env.coord_ligs = coord_ligs
        count += 1

        new_model = out_pose(n, new_model, quats, Ps)
        new_model.id = n+1
        struc_lig.add(copy.deepcopy(new_model))

        (edge_ints, send_ints, neigh_ints,
         env.intmask_recs, env.intmask_ligs,
         env.rimmask_recs, env.rimmask_ligs) = env.get_state(env.coord_ligs)

        if (n%500)==0:
            io.set_structure(copy.deepcopy(struc_lig))
            io.save('test_100ACT.pdb')

    io.set_structure(struc_lig)
    io.save('test_100ACT.pdb')


def check_edges(env):
    a = jnp.array([
        [1,0,0],[3,0,0],[5,0,0],
        [10,0,0],[20,0,0],[25,0,0],
        [45,0,0],[90,0,0],[200,0,0],
        [500,0,0],[501,0,0],[505,0,0],
        [0,0,0],[0,0,0],[0,0,0]])

    ma = jnp.array((1,1,1,1,1,1,1,1,1,1,1,1,0,0,0))

    b = jnp.array([
        [4,0,0],[8,0,0],[15,0,0],
        [30,0,0],[60,0,0],[120,0,0],
        [240,0,0],[360,0,0],[1060,0,0],
        [1061,0,0],[0,0,0],[0,0,0],
        [0,0,0],[0,0,0],[0,0,0]])

    mb = jnp.array((1,1,1,1,1,1,1,1,1,1,0,0,0,0,0))

    dmaps = vmap(distances_from_coords)(a[None,:,None,:], b[None,None,:,:])
    dmaps = dmaps*ma[None,:,None]*mb[None,None,:]
    dmaps = jnp.squeeze(dmaps)
    edges, nodes, neighs = env.get_edges(5, 12, dmaps)
    print (a)
    print (b)
    
    print (dmaps)

    print ('edge', edges)
    print ('nodes', nodes)
    print ('neigh', neighs)


def out_pose(name_id, struc_lig, quats, Ps):
    atom_objects = unfold_entities(struc_lig, 'A')
    atoms = jnp.array([atom.get_coord() for atom in atom_objects])
    
    atoms -= Ps
    atoms = quat_rotation(atoms, jnp.squeeze(quats))
    atoms += Ps

    idx = 0
    for atom in atom_objects:
            atom.set_coord(atoms[idx])
            idx += 1
    
    return struc_lig


if __name__ == "__main__":
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
    print (env.length_recs, env.length_ligs)

    (edge_ints, send_ints, neigh_ints,
     env.intmask_recs, env.intmask_ligs, 
     env.rimmask_recs, env.rimmask_ligs) = env.get_state(env.coord_ligs)

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

    #generate_random_sequence(key, struc_lig, env)

    ########################################
    # check edges
    ########################################

    check_edges(env)
