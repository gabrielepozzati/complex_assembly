import jax.numpy as jnp

def rot_from_pred(q):

    b, c, d = q / jnp.sqrt(jnp.sum(q**2)+1)
    a = 1 / jnp.sqrt(jnp.sum(q**2)+1)
    aa, bb, cc, dd = a**2, b**2, c**2, d**2

    return jnp.array(
        [[aa+bb-cc-dd,
          2*b*c-2*a*d,
          2*b*d+2*a*c],

         [2*b*c+2*a*d,
          aa-bb+cc-dd,
          2*c*d-2*a*b],

         [2*b*d-2*a*c,
          2*c*d+2*a*b,
          aa-bb-cc+dd]
        ])

def cmap_from_2D(matrix1, matrix2):
    return jnp.sqrt(jnp.sum((matrix1[:, None, :]-matrix2[None, :, :])**2, axis=-1))
