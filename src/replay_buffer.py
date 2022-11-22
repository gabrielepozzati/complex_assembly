import sys
import jax
import jax.numpy as jnp
import jax.random as jrn
from jax.tree_util import tree_map
from functools import partial
from jax import jit

class ReplayBuffer():
    def __init__(self, key, max_size, plist, enum, pad, device):
        self.key = key
        self.device = device
        self.actual_size = 0
        self.max_size = max_size
        self.pair_number = len(plist)
        self.pair_list = plist

        shape_idx = (self.max_size, 1, pad*enum*2)
        shape_edges = (self.max_size, 1, pad*enum*2, 2)
        shape_cloud = (self.max_size, 1, pad, 43)
        self.data = {
                'prev_nodes':jax.device_put(
                    jnp.empty(shape_cloud, dtype=jnp.float32),
                    device=device),

                'prev_edges':jax.device_put(
                    jnp.empty(shape_edges, dtype=jnp.float32), 
                    device=device),

                'prev_senders':jax.device_put(
                    jnp.empty(shape_idx, dtype=jnp.uint16), 
                    device=device),

                'prev_receivers':jax.device_put(
                    jnp.empty(shape_idx, dtype=jnp.uint16), 
                    device=device),

                'next_nodes':jax.device_put(
                    jnp.empty(shape_cloud, dtype=jnp.float32),
                    device=device),

                'next_edges':jax.device_put(
                    jnp.empty(shape_edges, dtype=jnp.float32), 
                    device=device),

                'next_senders':jax.device_put(
                    jnp.empty(shape_idx, dtype=jnp.uint16), 
                    device=device),

                'next_receivers':jax.device_put(
                    jnp.empty(shape_idx, dtype=jnp.uint16), 
                    device=device),

                'actions':jax.device_put(
                    jnp.empty((self.max_size, 1, 8), dtype=jnp.float32), 
                    device=device),

                'rewards':jax.device_put(
                    jnp.empty((self.max_size, 1), dtype=jnp.float32), 
                    device=device)}

        self.buffer = {code:self.data for code in plist}

        acc = (shape_idx[0]*shape_idx[1]*self.pair_number*2*4)
        acc += (shape_cloud[0]*shape_cloud[1]*self.pair_number*3*4)
        acc += (shape_edges[0]*shape_edges[1]*shape_edges[2]*self.pair_number*4*2)
        acc += (self.max_size*self.pair_number*8*4) + (self.max_size*self.pair_number*4)
        print (f'Buffer size {float(acc)/1e9} GB')

    @partial(jit, static_argnums=(0,))
    def add_to_replay(self, rbuffer, experience):
        def split_by_pair(idx, experience):
            prot_experience = {}
            for lbl in experience:
                array = experience[lbl][idx]
                prot_experience[lbl] = array
            return prot_experience

        def update_buffer(storage, experience):
            storage = storage[1:]
            experience = jnp.expand_dims(experience, axis=0)
            return jax.device_put(
                    jnp.concatenate((storage, experience[None,:]), axis=0), 
                    device=self.device)

        split_experience = {}
        for idx in range(self.pair_number):
            code = self.pair_list[idx]
            split_experience[code] = split_by_pair(idx, experience)

        return tree_map(update_buffer, rbuffer, split_experience), self.actual_size+1

    @partial(jit, static_argnums=(0,1))
    def sample_from_replay(self, batch_size, prot_buffers, key):

        def sample_batch(prot_buffer):
            bidxs = jrn.choice(bkey, self.max_size, shape=(batch_size,), replace=False)
            return tree_map(lambda x: x[bidxs], prot_buffer)

        key, bkey = jrn.split(key, 2)
        
        batch = sample_batch(prot_buffers[0])
        for prot_buffer in prot_buffers[1:]:
            key, bkey = jrn.split(key, 2)
            prot_batch = sample_batch(prot_buffer)
            batch = tree_map(lambda x, y: jnp.concatenate((x,y), axis=1), batch, prot_batch)

        return batch, key
