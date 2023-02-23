import os
import sys
import jax
import glob
import jraph
import pickle as pkl
from functools import partial
import jax.numpy as jnp
import pdb as debug
from jax import vmap, jit

def load_dataset(path, size=None, skip=0):
    count = 0
    dataset = {}
    print ('Set size:', size)
    for idx, path in enumerate(glob.glob(path+'/*')):
        if idx < skip: continue
        code = path.split('/')[-1].rstrip('.pkl')
        f = open(path, 'br')
        data = pkl.load(f)
        print (code)

        # restore device array type
        for lbl in ['coord', 'nodes', 'masks']:
            data[lbl] = (jnp.array(data[lbl][0]),jnp.array(data[lbl][1]))

        dataset[code] = data
        count += 1
        if count == size: break

    return dataset, code


@jax.jit
def rmsd(set1, set2, coord_num):
    residual_sum = jnp.sum(jnp.subtract(set1,set2)**2)
    return jnp.sqrt(residual_sum/coord_num)


@jax.jit
def distances_from_coords(coords1, coords2):
    return jnp.sqrt(jnp.sum((coords1-coords2)**2, axis=-1))


@jax.jit
def one_hot_distances(distances, bin_size=1):
    last = jnp.where(distances>32, 1, 0)
    bins = jnp.arange(32,0,-bin_size)
    one_hot = [jnp.where((distances<=n) & (distances>n-bin_size), 1, 0) for n in bins]
    return jnp.flip(jnp.stack([last]+one_hot, axis=-1), axis=-1)


def illegal_interfaces(dmaps, config, debug):
    dmaps = jnp.where(dmaps==0, 1e9, dmaps)
    distances = vmap(lambda x: jnp.min(x))(dmaps)
    
    if debug: print (distances)

    if jnp.any(distances > config['interface_threshold']) \
    or jnp.any(distances < config['clash_threshold']): return True
    else: return False


def get_illegal_idxs(dmaps, config):
    dmaps = jnp.where(dmaps==0, 1e9, dmaps)
    distances = vmap(lambda x: jnp.min(x))(dmaps)

    i_thr = config['interface_threshold']
    c_thr = config['clash_threshold']

    return jnp.argwhere((distances>i_thr)|(distances < c_thr))


def kabsch(dept, dest):
    '''
    Move dept. points to fit dest. equivalents as best as possible
    '''
    dept_mean = jnp.mean(dept, axis=0, keepdims=True)
    dest_mean = jnp.mean(dest, axis=0, keepdims=True)

    A = jnp.transpose(dest - dest_mean) @ (dept - dept_mean)
    U, S, Vt = jnp.linalg.svd(A)
    V = Vt.T
    U = U.T

    corr_mat = jnp.diag(jnp.array([1,1,jnp.sign(jnp.linalg.det(V@U))]))
    R = V @ (corr_mat @ U)
    t = dest_mean - jnp.transpose(R @ jnp.transpose(dept_mean))
    return R, jnp.squeeze(t)

def get_distances(min_dist, max_dist, c_rec, idxs, dist):
    '''
    given 3 values between 0 and 1 contained in dist, and a range 
    between min and max, map first value in said range, and map 
    remaining 2 values to get distances that allow precise trilateration 
    respective to the three points defined by c_rec[idxs]
    '''
    d1, d2, d3 = dist
    p1, p2, p3 = c_rec[idxs[2:]]
    d12 = jnp.linalg.norm(p1-p2)
    max_dist = jnp.maximum(max_dist, d12)

    # compute distance between p1 and ligand point
    d1l = (max_dist-min_dist)*d1+min_dist

    # compute lower distance limit btw p2 and ligand point
    min_d2l = jnp.abs((d1l-d12))

    # compute distance between p2 and ligand point
    d2l = (max_dist-min_d2l)*d2+min_d2l

    # compute center/radius/normal of intersection circle
    # btw sphere1 and sphere2
    d_p1_cc = 0.5*(d12+(d1l**2-d2l**2)/d12)
    cc = p1+d_p1_cc*(p2-p1)/d12
    cr = jnp.sqrt(d1l**2-d_p1_cc**2)
    cn = (p1-cc)/jnp.linalg.norm(p1-cc)

    # compute lower and upper distance limits btw p3 and ligand point
    delta = p3 - cc
    Ndelta = jnp.dot(cn, delta)
    crossNdelta = jnp.cross(cn,delta)
    radial_low = jnp.linalg.norm(crossNdelta) - cr
    radial_high = jnp.linalg.norm(crossNdelta) + cr
    min_d3l = jnp.sqrt(Ndelta**2+radial_low**2)
    max_d3l = jnp.sqrt(Ndelta**2+radial_high**2)

    # compute distance between p3 and ligand point
    d3l = (max_d3l-min_d3l)*d3+min_d3l

    return jnp.array([d1l, d2l, d3l])


def trilateration(c_rec, l_rec, idxs, dist):
    p1, p2, p3 = c_rec[idxs[2:]]
    r1, r2, r3 = dist

    e_x=(p2-p1)/jnp.linalg.norm(p2-p1)
    i=jnp.dot(e_x,(p3-p1))
    e_y=(p3-p1-(i*e_x))/(jnp.linalg.norm(p3-p1-(i*e_x)))
    e_z=jnp.cross(e_x,e_y)
    d=jnp.linalg.norm(p2-p1)
    j=jnp.dot(e_y,(p3-p1))
    x=((r1**2)-(r2**2)+(d**2))/(2*d)
    y=(((r1**2)-(r3**2)+(i**2)+(j**2))/(2*j))-((i/j)*(x))
    z1 = jnp.sqrt(r1**2-x**2-y**2)
    z2 = jnp.sqrt(r1**2-x**2-y**2)*(-1)
    l1 = p1+(x*e_x)+(y*e_y)+(z1*e_z)
    l2 = p1+(x*e_x)+(y*e_y)+(z2*e_z)

    c_rec_mean = jnp.sum(c_rec, axis=0)/l_rec
    d1 = jnp.linalg.norm(l1-c_rec_mean)
    d2 = jnp.linalg.norm(l2-c_rec_mean)
    select = jnp.argmax(jnp.array((d1,d2)))
    return jnp.array((l1, l2))[select]


def unfold_features(mask, nodes, edges, i, j):
    mask_rec, mask_lig = jnp.split(mask, 2, axis=0)
    padmask_rec, intmask_rec, rimmask_rec = mask_rec[:,0], mask_rec[:,1], mask_rec[:,2]
    padmask_lig, intmask_lig, rimmask_lig = mask_lig[:,0], mask_lig[:,1], mask_rec[:,2]

    nodes_rec, nodes_lig = jnp.split(nodes, 2, axis=0)

    edges_intra, edges_int = jnp.split(edges, 2, axis=0)
    i_intra, i_int = jnp.split(i, 2, axis=0)
    j_intra, j_int = jnp.split(j, 2, axis=0)

    edges_rec, edges_lig = jnp.split(edges_intra, 2, axis=0)
    i_rec, i_lig = jnp.split(i_intra, 2, axis=0)
    j_rec, j_lig = jnp.split(j_intra, 2, axis=0)
    
    return (padmask_rec, padmask_lig, intmask_rec, intmask_lig, 
            rimmask_rec,  rimmask_lig, nodes_rec, nodes_lig, 
            edges_rec, edges_lig, edges_int, 
            i_rec, i_lig, i_int, 
            j_rec, j_lig, j_int)


### TODO WRITE AS A CLASS
def write_mmcif(chain_ids, transforms, in_path, out_path) -> None:
    io = MMCIFIO()
    pdbp = PDBParser()
    cifp = MMCIFParser()
    struc1 = pdbp.get_structure('', in_path[0])
    struc2 = pdbp.get_structure('', in_path[1])
    
    if os.path.exists(out_path):
        out_struc = cifp.get_structure('-', out_path)
        for model in out_struc: 
            last_model_id = model.get_id() + 1
        new_model_id = last_model_id+1
    else:
        out_struc = Structure('-')
        new_model_id = 1

    out_model = Model(new_model_id)
    out_struc.add(out_model)
    out_model.set_parent(out_struc)

    for rot, tr in transforms:
        mid, cid = ids.pop(0)
        if mid in struc1:
            if cid in struc1[mid]: in_chain = struc1[mid][cid]
        if mid in struc2:
            if cid in struc2[mid]: in_chain = struc2[mid][cid]
        cloud = [atom.get_coord() for atom in unfold_entities(in_chain, 'A')]
        cloud = jnp.array(cloud)
        cloud = jnp.matmul(rot, cloud.transpose()).transpose() - tr
        for new, atom in zip(cloud, unfold_entities(in_chain, 'A')): 
            atom.set_coord(new)

        in_chain.set_parent(out_model)
        out_model.add(in_chain)

    for chain in unfold_entities(out_struc, 'C'):
        print (chain.get_full_id())

    io.set_structure(out_struc)
    io.save(out_path)


def save_to_model(out_struc, ref_model, quat, tr, init=False):
    atoms = unfold_entities(ref_model, 'A')
    cloud = jnp.array([atom.get_coord() for atom in atoms])

    if init:
        cloud = quat_rotation(cloud-tr, quat)
    else:
        cloud_cm = jnp.mean(cloud, axis=0)
        cloud = quat_rotation(cloud-cloud_cm, quat)+cloud_cm
        cloud += tr

    model_num = len(unfold_entities(out_struc, 'M'))
    model = copy.deepcopy(ref_model)
    model.detach_parent()
    model.id = model_num+1
    model.set_parent(out_struc)
    out_struc.add(model)

    atoms = unfold_entities(model, 'A')
    for xyz, atom in zip(cloud, atoms): atom.set_coord(xyz)

    return out_struc
