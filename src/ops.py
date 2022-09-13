import jax
import jax.numpy as jnp

def matrix_from_quat(q):
    a, b, c, d = q
    aa, bb, cc, dd = a**2, b**2, c**2, d**2

    return jnp.array(
        [[aa+bb-cc-dd, 2*b*c-2*a*d, 2*b*d+2*a*c],
         [2*b*c+2*a*d, aa-bb+cc-dd, 2*c*d-2*a*b],
         [2*b*d-2*a*c, 2*c*d+2*a*b, aa-bb-cc+dd]])

def cmap_from_cloud(matrix1, matrix2):
    return jnp.sqrt(jnp.sum((matrix1-matrix2)**2, axis=-1))

def one_hot_cmap(cmap):
    bin_size = 1
    if len(cmap.shape) == 1: cmap = jnp.expand_dims(cmap, axis=-1)
    ones, zeros = jnp.ones(cmap.shape), jnp.zeros(cmap.shape)
    bmap = jnp.where(cmap>32, ones, zeros)
    for n in range(32, 0, -bin_size):
        next_bin = jnp.where((cmap<=n) & (cmap>n-bin_size), ones, zeros)
        bmap = jnp.concatenate([bmap, zeros], axis=-1)
    
    return bmap

def quat_rotation(cloud, q, substeps):
    #quaternion coordinates
    zeros = jnp.zeros((cloud.shape[0], 1))
    p = jnp.concatenate((zeros, cloud), axis=-1)
    p = jnp.reshape(hamilton_multiplier(p), (cloud.shape[0], 4, 4))

    #inverse quaternion
    q_i = jnp.concatenate((q[:,:1], -q[:,1:]), axis=-1)
    q_i = jnp.reshape(hamilton_multiplier(q_i), (substeps, 4, 4))

    qp = quat_product(q[:,None,None,:], p[None,:,:,:])
    return quat_product(qp[:,:,None,:], q_i[:,None,:,:])[:,:,1:]

def hamilton_multiplier(q):
    return jnp.array([
        [q[:,0],-q[:,1],-q[:,2],-q[:,3]],
        [q[:,1],q[:,0],-q[:,3],q[:,2]],
        [q[:,2],q[:,3],q[:,0],-q[:,1]],
        [q[:,3],-q[:,2],q[:,1],q[:,0]]])

def quat_product(q1, q2):
    return jnp.sum(q1*q2, axis=-1)

