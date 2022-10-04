import copy
import jax
import jax.numpy as jnp
import pickle

from Bio.PDB.Structure import Structure
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Polypeptide import three_to_one
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Selection import unfold_entities

H0_MASK = jnp.array(
    [[ 1, 0, 0, 0],
     [ 0, 1, 0, 0],
     [ 0, 0, 1, 0],
     [ 0, 0, 0, 1]])

H1_MASK = jnp.array(
    [[ 0,-1, 0, 0],
     [ 1, 0, 0, 0],
     [ 0, 0, 0,-1],
     [ 0, 0, 1, 0]])

H2_MASK = jnp.array(
    [[ 0, 0,-1, 0],
     [ 0, 0, 0, 1],
     [ 1, 0, 0, 0],
     [ 0,-1, 0, 0]])

H3_MASK = jnp.array(
    [[ 0, 0, 0,-1],
     [ 0, 0,-1, 0],
     [ 0, 1, 0, 0],
     [ 1, 0, 0, 0]])

H_MASK = jnp.stack(
    (H0_MASK, H1_MASK,
     H2_MASK, H3_MASK))

def matrix_from_quat(q):
    a, b, c, d = q
    aa, bb, cc, dd = a**2, b**2, c**2, d**2

    return jnp.array(
        [[aa+bb-cc-dd, 2*b*c-2*a*d, 2*b*d+2*a*c],
         [2*b*c+2*a*d, aa-bb+cc-dd, 2*c*d-2*a*b],
         [2*b*d-2*a*c, 2*c*d+2*a*b, aa-bb-cc+dd]])

def cmap_from_cloud(matrix1, matrix2):
    return jnp.sqrt(jnp.sum((matrix1-matrix2)**2, axis=-1))

def one_hot_cmap(cmap, bin_size=1):
    ones, zeros = jnp.ones(cmap.shape), jnp.zeros(cmap.shape)
    bmap = jnp.where(cmap>32, 1, 0)
    for n in range(32, 0, -bin_size):
        next_bin = jnp.where((cmap<=n) & (cmap>n-bin_size), 1, 0)
        bmap = jnp.concatenate([next_bin, bmap], axis=-1)
    
    return bmap

def quat_rotation(cloud, q, substeps):
    #quaternion coordinates
    zeros = jnp.zeros((cloud.shape[0], 1))
    p = jnp.concatenate((zeros, cloud), axis=-1)
    p = hamilton_multiplier(p)

    #inverse quaternion
    q_i = jnp.concatenate((q[:,:1], -q[:,1:]), axis=-1)
    q_i = hamilton_multiplier(q_i)

    qp = quat_product(q[:,None,None,:], p[None,:,:,:])
    return quat_product(qp[:,:,None,:], q_i[:,None,:,:])[:,:,1:]

def hamilton_multiplier(q):
    return jnp.sum(q[:,:,None,None]*H_MASK[None,:,:,:], axis=1)

def quat_product(q1, q2):
    return jnp.sum(q1*q2, axis=-1)

def quat_composition(q1, q2):
    q2 = jnp.sum(q2[:,None,None]*H_MASK[:,:,:], axis=0)
    return jnp.sum(q1[None,:]*q2, axis=-1)

def main():
    q_I = jnp.array([1.,0.,0.,0.])
    q_x90 = jnp.array([0.7071,0.7071,0.,0.])
    q_y90 = jnp.array([0.7071,0.,0.7071,0.])
    q_z90 = jnp.array([0.7071,0.,0.,0.7071])

    data_path = '/home/pozzati/complex_assembly/data/train_features.pkl'
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)

    io = PDBIO()
    pdbp = PDBParser(QUIET=True)
    lpath = '/home/pozzati/complex_assembly/data/benchmark5.5/1A2K_l_u.pdb'
    struc = pdbp.get_structure('', lpath)
    atoms = unfold_entities(struc, 'A')
    cloud = jnp.array([atom.get_coord() for atom in atoms])
    cloud -= (jnp.sum(cloud, axis=0)/cloud.shape[0])[None,:]
    
    q_x90I = jnp.concatenate((q_x90[:1], -q_x90[1:]), axis=-1)
    q_y90I = jnp.concatenate((q_y90[:1], -q_y90[1:]), axis=-1)
  
    q_xy90 = quat_composition(q_x90, q_y90)
    q_xy90I = quat_composition(q_y90I, q_x90I)
    print (q_xy90, q_xy90I)

    action = jnp.array([q_I,q_x90,q_y90,q_z90, q_xy90])
    rot = quat_rotation(cloud, action, 5)
    
    out_struc = Structure('test')
    
    nextmodel = copy.deepcopy(struc[0])
    nextmodel.detach_parent()
    atoms = unfold_entities(nextmodel, 'A')
    for xyz, atom in zip(cloud, atoms):
        atom.set_coord(xyz)
    out_struc.add(nextmodel)
    nextmodel.set_parent(out_struc)

    for idx, step in enumerate(rot):
        nextmodel = copy.deepcopy(struc[0])
        nextmodel.detach_parent()

        atoms = unfold_entities(nextmodel, 'A')
        for xyz, atom in zip(step, atoms): 
            atom.set_coord(xyz)
        nextmodel.id = idx+1
        out_struc.add(nextmodel)
        nextmodel.set_parent(out_struc)

    io.set_structure(out_struc)
    io.save('test.pdb')

if __name__ == '__main__':
    main()
