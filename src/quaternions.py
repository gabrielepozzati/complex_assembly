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

def quat_from_pred(array):
    '''
    given array of shape (3,) containing values between 0 and 1 
    define a corresponding quaternion
    '''

    ndims = len(array.shape)
    if ndims == 1: array = jnp.expand_dims(array, axis=0)

    norm_factor = jnp.sqrt(1+jnp.sum(array**2, axis=-1))
    norm_factor = jnp.expand_dims(norm_factor, axis=1)

    ones_shape = (array.shape[0], 1)
    array = jnp.concatenate((jnp.ones(ones_shape), array), axis=-1)
    return array/norm_factor


def quat_from_pivoting(P, p0, p1):
    '''
    P = pivot coordinates
    p0 = starting point coordinates
    p1 = ending point coordinates
    '''

    p0 = (p0-P)/jnp.linalg.norm(p0-P)
    p1 = (p1-P)/jnp.linalg.norm(p1-P)

    dot = jnp.dot(p0, p1)
    axis = jnp.cross(p0, p1)
    axis /= jnp.linalg.norm(axis)
    angle = jnp.clip(jnp.arccos(dot), a_min=0, a_max=jnp.pi/12)
    s = jnp.sin(angle/2)
    return jnp.array((jnp.cos(angle/2), axis[0]*s, axis[1]*s, axis[2]*s))


def matrix_from_quat(q):
    a, b, c, d = q
    aa, bb, cc, dd = a**2, b**2, c**2, d**2
    return jnp.array(
        [[aa+bb-cc-dd, 2*b*c-2*a*d, 2*b*d+2*a*c],
         [2*b*c+2*a*d, aa-bb+cc-dd, 2*c*d-2*a*b],
         [2*b*d-2*a*c, 2*c*d+2*a*b, aa-bb-cc+dd]])


def quat_rotation(cloud, q):
    #quaternion coordinates
    zeros = jnp.zeros((cloud.shape[0], 1))
    p = jnp.concatenate((zeros, cloud), axis=-1)
    p = jax.vmap(hamilton_multiplier, in_axes=0, out_axes=0)(p)
    
    #inverse quaternion
    q_i = jnp.concatenate((q[:1], -q[1:]), axis=-1)
    q_i = hamilton_multiplier(q_i)
    
    qp = quat_product(q[None,None,:], p[:,:,:])
    return quat_product(qp[:,None,:], q_i[None,:,:])[:,1:]


def hamilton_multiplier(q):
    return jnp.sum(q[:,None,None]*H_MASK[:,:,:], axis=-3)


def quat_product(q1, q2):
    return jnp.sum(q1*q2, axis=-1)


def quat_composition(q1, q2):
    q2 = jnp.sum(q2[:,None,None]*H_MASK[:,:,:], axis=0)
    return jnp.sum(q1[None,:]*q2, axis=-1)


def test():
    q_I = jnp.array([1.,0.,0.,0.])
    q_x90 = jnp.array([0.7071,0.7071,0.,0.])
    q_y90 = jnp.array([0.7071,0.,0.7071,0.])
    q_z90 = jnp.array([0.7071,0.,0.,0.7071])

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

    out_struc = Structure('test')
    nextmodel = copy.deepcopy(struc[0])
    nextmodel.detach_parent()
    atoms = unfold_entities(nextmodel, 'A')
    for xyz, atom in zip(cloud, atoms): atom.set_coord(xyz)
    out_struc.add(nextmodel)
    nextmodel.set_parent(out_struc)

    actions = jnp.array([q_x90,q_y90,q_z90, q_xy90])
    for idx, action in enumerate(actions):
        cloud = quat_rotation(cloud, action)
    
        nextmodel = copy.deepcopy(struc[0])
        nextmodel.detach_parent()

        atoms = unfold_entities(nextmodel, 'A')
        for xyz, atom in zip(cloud, atoms): atom.set_coord(xyz)
        nextmodel.id = idx+1
        out_struc.add(nextmodel)
        nextmodel.set_parent(out_struc)

    io.set_structure(out_struc)
    io.save('test.pdb')

if __name__ == '__main__':
    test()

