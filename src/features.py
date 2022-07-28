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
from coordinates import *

import matplotlib.pyplot as plt


standard_residues = [
    'A','R','N','D','C','Q','E','G','H','I',
    'L','K','M','F','P','S','T','W','Y','V']

standard_residues_three = [
    'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']

max_asa = {
    'ALA':129.0,'ARG':274.0,'ASN':195.0,'ASP':193.0,'CYS':167.0,
    'GLN':225.0,'GLU':223.0,'GLY':104.0,'HIS':224.0,'ILE':197.0,
    'LEU':201.0,'LYS':236.0,'MET':224.0,'PHE':240.0,'PRO':159.0,
    'SER':155.0,'THR':172.0,'TRP':285.0,'TYR':263.0,'VAL':174.0}

atom_types = {'ALA':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1]},
              'ARG':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'NE':[0,0,0,1,0,0,0,0,0,0,0],
                     'NH1':[0,0,0,1,0,0,0,0,0,0,0],
                     'NH2':[0,0,0,1,0,0,0,0,0,0,0],
                      'CZ':[0,0,0,0,0,0,0,0,1,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                      'CG':[0,0,0,0,0,0,0,0,0,0,1],
                      'CD':[0,0,0,0,0,0,0,0,0,0,1]},
              'ASN':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                     'ND2':[0,1,0,0,0,0,0,0,0,0,0],
                     'OD1':[0,0,0,0,0,1,0,0,0,0,0],
                      'CG':[0,0,0,0,0,0,0,0,1,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1]},
              'ASP':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                     'OD1':[0,0,0,0,0,0,0,1,0,0,0],
                     'OD2':[0,0,0,0,0,0,0,1,0,0,0],
                      'CG':[0,0,0,0,0,0,0,0,1,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1]},
              'CYS':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'SG':[1,0,0,0,0,0,0,0,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1]},
              'GLN':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                     'NE2':[0,1,0,0,0,0,0,0,0,0,0],
                     'OE1':[0,0,0,0,0,1,0,0,0,0,0],
                      'CD':[0,0,0,0,0,0,0,0,1,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                      'CG':[0,0,0,0,0,0,0,0,0,0,1]},
              'GLU':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                     'OE1':[0,0,0,0,0,0,0,1,0,0,0],
                     'OE2':[0,0,0,0,0,0,0,1,0,0,0],
                      'CD':[0,0,0,0,0,0,0,0,1,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                      'CG':[0,0,0,0,0,0,0,0,0,0,1]},
              'GLY':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0]},
              'HIS':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                     'ND1':[0,0,1,0,0,0,0,0,0,0,0],
                     'NE2':[0,0,1,0,0,0,0,0,0,0,0],
                      'CG':[0,0,0,0,0,0,0,0,0,1,0],
                     'CD2':[0,0,0,0,0,0,0,0,0,1,0],
                     'CE1':[0,0,0,0,0,0,0,0,0,1,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1]},
              'ILE':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                     'CG1':[0,0,0,0,0,0,0,0,0,0,1],
                     'CG2':[0,0,0,0,0,0,0,0,0,0,1],
                     'CD1':[0,0,0,0,0,0,0,0,0,0,1]},
              'LEU':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                      'CG':[0,0,0,0,0,0,0,0,0,0,1],
                     'CD1':[0,0,0,0,0,0,0,0,0,0,1],
                     'CD2':[0,0,0,0,0,0,0,0,0,0,1]},
              'LYS':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'NZ':[0,0,0,0,1,0,0,0,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                      'CG':[0,0,0,0,0,0,0,0,0,0,1],
                      'CD':[0,0,0,0,0,0,0,0,0,0,1],
                      'CE':[0,0,0,0,0,0,0,0,0,0,1]},
              'MET':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                      'CG':[0,0,0,0,0,0,0,0,0,0,1],
                      'CE':[0,0,0,0,0,0,0,0,0,0,1],
                      'SD':[1,0,0,0,0,0,0,0,0,0,0]},
              'PHE':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                      'CG':[0,0,0,0,0,0,0,0,0,1,0],
                     'CD1':[0,0,0,0,0,0,0,0,0,1,0],
                     'CD2':[0,0,0,0,0,0,0,0,0,1,0],
                     'CE1':[0,0,0,0,0,0,0,0,0,1,0],
                     'CE2':[0,0,0,0,0,0,0,0,0,1,0],
                      'CZ':[0,0,0,0,0,0,0,0,0,1,0]},
              'PRO':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                      'CG':[0,0,0,0,0,0,0,0,0,0,1],
                      'CD':[0,0,0,0,0,0,0,0,0,0,1]},
              'SER':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'OG':[0,0,0,0,0,0,1,0,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1]},
              'THR':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                     'OG1':[0,0,0,0,0,0,1,0,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                     'CG2':[0,0,0,0,0,0,0,0,0,0,1]},
              'TRP':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                      'CG':[0,0,0,0,0,0,0,0,0,1,0],
                     'CD1':[0,0,0,0,0,0,0,0,0,1,0],
                     'CD2':[0,0,0,0,0,0,0,0,0,1,0],
                     'CE1':[0,0,0,0,0,0,0,0,0,1,0],
                     'CE2':[0,0,0,0,0,0,0,0,0,1,0],
                     'CE3':[0,0,0,0,0,0,0,0,0,1,0],
                     'CZ2':[0,0,0,0,0,0,0,0,0,1,0],
                     'CZ3':[0,0,0,0,0,0,0,0,0,1,0],
                     'CH2':[0,0,0,0,0,0,0,0,0,1,0],
                     'NE1':[0,0,1,0,0,0,0,0,0,0,0]},
              'TYR':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'CG':[0,0,0,0,0,0,0,0,0,1,0],
                     'CD1':[0,0,0,0,0,0,0,0,0,1,0],
                     'CD2':[0,0,0,0,0,0,0,0,0,1,0],
                     'CE1':[0,0,0,0,0,0,0,0,0,1,0],
                     'CE2':[0,0,0,0,0,0,0,0,0,1,0],
                      'CZ':[0,0,0,0,0,0,0,0,0,1,0],
                      'OH':[0,0,0,0,0,0,1,0,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1]},
              'VAL':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                     'CG1':[0,0,0,0,0,0,0,0,0,0,1],
                     'CG2':[0,0,0,0,0,0,0,0,0,0,1]}}

def one_hot_cmap(cmap):
    bin_size = 1
    ones, zeros = jnp.ones(cmap.shape), jnp.zeros(cmap.shape)
    bmap = jnp.where(cmap>32, ones, zeros)
    for n in range(32, 0, -bin_size):
        next_bin = jnp.where((cmap<=n) & (cmap>n-bin_size), ones, zeros)
        bmap = jnp.concatenate([bmap, next_bin], axis=-1)
    return jnp.concatenate([bmap, zeros], axis=-1)


def format_data(pdb_paths: list, out_path: str):
    #cifp = MMCIFParser()
    cifp = PDBParser(QUIET=True)
    sasa = ShrakeRupley()
 
    dataset = {}
    for path in pdb_paths:
        print (path)
        pdb = path.split('/')[-1][0:4]
        path2 = path[:-7]+'l'+path[-6:]
        structure1 = cifp.get_structure(pdb, path)
        structure2 = cifp.get_structure(pdb, path2)
        dataset[pdb] = {}

        clouds = []
        graphs = []
        surf_masks = []
        chains1 = unfold_entities(structure1, 'C')
        chains2 = unfold_entities(structure2, 'C')
        chains = chains1+chains2
        for chain in chains:
            sasa.compute(chain, level='R')
            sasa.compute(chain, level='A')
            cid = chain.get_id()
           
            cloud = []
            nodes = []
            surf_mask = []
            for residue in chain: 
                rid = residue.get_resname()
                if rid not in standard_residues_three: continue

                # get CA coordinates to compute edges
                if 'CA' not in residue: continue
                x, y, z = residue['CA'].get_coord()
                cloud.append([x, y, z])

                # get residue RSA
                surf_mask.append(residue.sasa/max_asa[rid])

                # get atom types/ASA to compute node features
                atoms = []
                for atom in residue:
                    aid = atom.get_id()
                    atoms.append(atom_types[rid][aid]+[atom.sasa])
                while len(atoms) < 16: atoms.append(12*[0])
                nodes.append(atoms)
        
            # store point cloud and surface mask
            cloud = jnp.array(cloud)
            clouds.append(cloud)
            surf_mask = jnp.array(surf_mask)
            surf_masks.append(surf_mask)

            # compute all vs all cmap
            cmap = jnp.sqrt(jnp.sum(
                (cloud[:, None, :]-cloud[None, :, :])**2, axis=-1))

            bmap = one_hot_cmap(cmap)

            # compute edges and edges features
            sidx, ridx = jnp.indices(cmap.shape)
            sidx, ridx = jnp.ravel(sidx), jnp.ravel(ridx)
            edges = bmap[sidx,ridx]
            
            # put together chain graph and store it
            graph = jraph.GraphsTuple(
                nodes=jnp.array(nodes), edges=jnp.array(edges), 
                senders=jnp.array(sidx), receivers=jnp.array(ridx),
                n_node=jnp.array([len(nodes)]), n_edge=jnp.array([len(edges)]), 
                globals=jnp.array([1]*8))
            graphs.append(graph)

        # store chain labels
        clabels = [[idx] for idx, chain in enumerate(chains)]

        # compute graph of real interfaces
        sidx, ridx, edges = [], [], []
        nodes = [idx for idx, chain in enumerate(chains)]
        all_pairs = [[i,j] for i in range(len(nodes)) for j in range(len(nodes))]
        for i, j in all_pairs:
            if i >= j: continue
            cmap = jnp.sqrt(jnp.sum(
                (clouds[i][:, None, :]-clouds[j][None, :, :])**2, axis=-1))

            if not jnp.any(cmap < 8): continue
            edges += [cmap, jnp.transpose(cmap)]
            ridx += [j, i]
            sidx += [i, j]

        igraph =  jraph.GraphsTuple(
                nodes=nodes, senders=sidx, receivers=ridx,
                edges=edges, n_node=len(nodes), n_edge=len(edges),
                globals=None)
        
        clouds, tgroup = initialize_clouds(clouds, 42)

        dataset[pdb]['clouds'] = clouds
        dataset[pdb]['graphs'] = graphs
        dataset[pdb]['igraph'] = igraph
        dataset[pdb]['clabel'] = clabels
        dataset[pdb]['smask'] = surf_mask

        #print (tgroup)
        #print (id_mask)
        #in_paths = [path, path2]
        #write_mmcif(igraph, tgroup, in_paths, '/proj/berzelius-2021-29/users/x_gabpo/complex_assembly/test.cif')
        #sys.exit()

    with open(out_path, 'wb') as out:
        pickle.dump(dataset, out)


if __name__ == '__main__':
    datapath = '/home/pozzati/complex_assembly/data/benchmark5.5/*_r_b.pdb'
    path_list = [line.rstrip() for line in glob.glob(datapath)][:10]
    format_data(path_list, '/home/pozzati/complex_assembly/data/train_features.pkl')
