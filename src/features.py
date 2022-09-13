import os
import sys
import glob
import jraph
import pickle
import jax.numpy as jnp
import pdb as debug


from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Polypeptide import three_to_one
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Selection import unfold_entities

from Bio import pairwise2
from Bio.Align import substitution_matrices
blosum62 = substitution_matrices.load("BLOSUM62")

from coordinates import *
from tables import *

import matplotlib.pyplot as plt


def format_data(chains, mapping):
    masks, nodes, clouds = [], [], [] 
    for key in mapping:
        if type(mapping[key]) == int: continue
        chain = chains[key]
        sasa = ShrakeRupley()
        sasa.compute(chain, level='R')
        sasa.compute(chain, level='A')
        cid = chain.get_id()
       
        mask, node, cloud = [], [], []
        for residue in chain: 
            rid = residue.get_resname()
            if rid not in standard_residues_three: continue

            # skip residues without CA
            if 'CA' not in residue: continue

            # get residue RSA
            mask.append(residue.sasa/max_asa[rid])

            # get CA coordinates to compute edges
            x, y, z = residue['CA'].get_coord()
            cloud.append([x, y, z])

            # get atom types/ASA to compute node features
            atoms = []
            for atom in residue:
                aid = atom.get_id()
                atoms.append(atom_types[rid][aid]+[atom.sasa])
            while len(atoms) < 16: atoms.append(12*[0])
            node.append(atoms)
        
        # store point cloud and surface mask
        masks.append(jnp.array(mask))
        nodes.append(jnp.array(node))
        clouds.append(jnp.array(cloud))
    
    clouds, tgroup = initialize_clouds(clouds, 42)
    
    # pad to the same length
    max_len = max([len(mask_seq) for mask_seq in masks])
    for idx in range(len(masks)):
        pad = max_len - masks[idx].shape[0]
        masks[idx] = jnp.concatenate(
            [masks[idx], -jnp.ones([pad])], axis=0)
        nodes[idx] = jnp.concatenate(
            [nodes[idx], jnp.zeros([pad, 16, 12])], axis=0)
        clouds[idx] = jnp.concatenate(
            [clouds[idx], jnp.zeros([pad, 3])], axis=0)

    masks = jnp.array(masks)
    nodes = jnp.array(nodes)
    clouds = jnp.array(clouds)

    # compute graph of real interfaces
    sidx, ridx, edges_i = [], [], []
    nodes_i = [i for i in range(len(chains))]
    all_pairs = [[i,j] for i in nodes_i for j in nodes_i]
    for i, j in all_pairs:
        if i >= j: continue
        cloudi = jnp.array([res['CA'].get_coord() for res in chains[i] \
            if 'CA' in res and residue.get_resname() in standard_residues_three])
        cloudj = jnp.array([res['CA'].get_coord() for res in chains[j] \
            if 'CA' in res and residue.get_resname() in standard_residues_three])
        cmap = jnp.sqrt(jnp.sum((cloudi[:, None, :]-cloudj[None, :, :])**2, axis=-1))
        
        if not jnp.any(cmap < 8): continue

        edges_i += [cmap, jnp.transpose(cmap)]
        ridx += [j, i]
        sidx += [i, j]

    igraph = jraph.GraphsTuple(
            nodes=nodes_i, edges=edges_i,
            senders=sidx, receivers=ridx,
            n_node=len(nodes_i), n_edge=len(edges_i),
            globals=None)
        
    return {'masks' : masks, 'nodes' : nodes, 'clouds' : clouds, 
            'mapping' : mapping, 'igraph' : igraph}
    

def compare_chains(chains):
    mapping = {}
    for nkey, chain in enumerate(chains):
        seq = ''.join([three_to_one(res.get_resname()) for res in chain])

        match = None
        for okey in mapping:
            if type(mapping[okey]) == int: continue
            ali = pairwise2.align.globalds(seq, mapping[okey][0], blosum62, -11, -1)
            ali = [str(ali[0][0]), str(ali[0][1])] 

            while ali[0][0] == '-' or ali[1][0] == '-':
                ali[0], ali[1] = ali[0][1:], ali[1][1:]
            while ali[0][-1] == '-' or ali[1][-1] == '-':
                ali[0], ali[1] = ali[0][:-1], ali[1][:-1]
            seqid = sum([1. if a==b else 0. for a, b in zip(ali[0], ali[1])])
            seqid /= len(ali[0])

            if seqid >= 0.95: match = okey

        if match:
            if len(seq) > len(mapping[match][0]): 
                mapping[nkey] = [seq, [chain.get_full_id()]+mapping[match][1]]
                mapping[match] = nkey
            else: 
                mapping[match][1].append(chain.get_full_id())
                mapping[nkey] = match
        else: mapping[nkey] = [seq, [chain.get_full_id()]]

    return mapping


def main():
    pdbp = PDBParser(QUIET=True)
    cifp = MMCIFParser(QUIET=True)

    data_path = '/home/pozzati/complex_assembly/data/benchmark5.5/*_r_b.pdb'
    output_path = '/home/pozzati/complex_assembly/data/train_features.pkl'
    path_list = [line.rstrip() for line in glob.glob(data_path)][:10]
    
    dataset = {}
    for path in path_list:
        print (path)
        pdb = path.split('/')[-1][0:4]
        path2 = path[:-7]+'l'+path[-6:]
        structure1 = pdbp.get_structure(pdb, path)
        structure2 = pdbp.get_structure(pdb, path2)

        chains1 = unfold_entities(structure1, 'C')
        chains2 = unfold_entities(structure2, 'C')
        chains = chains1+chains2

        mapping = compare_chains(chains)

        dataset[pdb] = format_data(chains, mapping)
    
    with open(output_path, 'wb') as out:
        pickle.dump(dataset, out)

if __name__ == '__main__':
    main()
