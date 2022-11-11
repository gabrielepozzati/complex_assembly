import jax
import jax.numpy as jnp
import jax.random as jrn

class ReplayBuffer(key, max_size, pnum, enum, pad):
    def __init__(self):
        self.key = key
        self.actual_size = 0
        self.max_size = max_size
        
        shape_idx = (1, pnum, pad*enum)
        shape_edges = (1, pnum, pad*enum, 2)
        self.data = {
                'prev_edges':jnp.empty(shape_edges, dtype=jnp.float32),
                'prev_senders':jnp.empty(shape_idx, dtype=jnp.uint16),
                'prev_receivers':jnp.empty(shape_idx, dtype=jnp.uint16),
                'next_edges':jnp.empty(shape_edges, dtype=jnp.float32),
                'next_senders':jnp.empty(shape_idx, dtype=jnp.uint16),
                'next_receivers':jnp.empty(shape_idx, dtype=jnp.uint16),
                'actions':jnp.empty((1, pnum, 8), dtype=jnp.float32),
                'rewards':jnp.empty((1, pnum), dtype=jnp.float32)}

    def add_to_replay(self, experience):
        if actual_size == self.max_size:
            for key in data: data[key] = data[key][1:]
        else: actual_size += 1

        for idx, (key, array) in enumerate(self.data.items()):
            self.data[key] = jnp.concatenate((array, experience[idx]), axis=0)

    def sample_from_replay(self, batch_size):
        self.key, key = jrn.split(self.key)
        idxs = jrn.choice(key, self.actual_size, shape=(batch_size,), replace=False)

        return {key:data[key][idxs] for key in data}


