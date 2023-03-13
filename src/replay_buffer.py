import sys
import jax
import numpy as np
import jax.numpy as jnp
import jax.random as jrn
from jax.tree_util import tree_map
from functools import partial
from jax import jit, vmap

class ReplayBuffer():
    def __init__(self, config, plist):
        
        

        self.pair_list = plist
        self.pair_number = len(plist)
        
        self.pad = config['padding']
        self.enum = config['edge_number']
        
        self.buff_actual = 0
        self.cache_actual = 0
        self.bsn = config['batch_size_num']
        self.bsp = config['batch_size_pair']
        self.buff_size = config['buffer_size']
        self.cache_size = config['cache_size']

        assert self.cache_size < self.buff_size and self.buff_size%self.cache_size == 0, \
        'Buffer size must be larger and divisible by cache size'

        self.cache_shape = (self.pair_number, self.cache_size)
        self.cache = self.get_datastructure(self.cache_shape)
        self.cache_number = int(self.buff_size/self.cache_size)
        self.cache = jax.device_put(self.cache, device=jax.devices('gpu')[0])

        self.buff_shape = (self.pair_number, self.buff_size)
        self.buff = self.get_datastructure(self.buff_shape)

        p_num = self.pair_number
        i_num = p_num*self.pad*self.enum*2*2*2
        e_num = p_num*self.pad*self.enum*2*15*2*4
        m_num = p_num*self.pad*2*2*2
        a_num = p_num*3*self.pad*4
        r_num = p_num*4

        tot_ex = i_num+e_num+m_num+a_num+r_num
        tot_b = tot_ex * self.buff_size
        tot_c = tot_ex * self.cache_size
        print (f'Number of examples in cache:', self.cache_size)
        print (f'Number of examples in buffer:', self.buff_size)
        print (f'Size of 1 example per pair: {round((tot_ex/1e9),3)}GB')
        print (f'Cache size: {round((tot_c/1e9),3)}GB')
        print (f'Buffer size: {round((tot_b/1e9),3)}GB')


    def get_datastructure(self, common_shape):

        pad, enum = self.pad, self.enum
        idxs_shape = common_shape + (pad*enum*2,)
        edges_shape = common_shape + (pad*enum*2, 15)
        masks_shape = common_shape + (pad*2, 2)
        actions_shape = common_shape + (3, pad)

        def _get_placeholders():
            masks = jax.device_put(
                    jnp.empty(masks_shape, dtype=jnp.uint8), device=jax.devices('cpu')[0])

            edges = jax.device_put(
                    jnp.empty(edges_shape, dtype=jnp.float32), device=jax.devices('cpu')[0])

            idxs = jax.device_put(
                    jnp.empty(idxs_shape, dtype=jnp.uint16), device=jax.devices('cpu')[0])

            return masks, edges, idxs, idxs

        (prev_masks, prev_edges,
         prev_sends, prev_neighs) = _get_placeholders()

        (next_masks, next_edges,
         next_sends, next_neighs) = _get_placeholders()

        actions = jax.device_put(
                jnp.empty(actions_shape, dtype=jnp.float32), device=jax.devices('cpu')[0])

        rewards = jax.device_put(
                jnp.empty(common_shape, dtype=jnp.float32), device=jax.devices('cpu')[0])

        return [prev_masks, prev_edges, prev_sends, prev_neighs,
                next_masks, next_edges, next_sends, next_neighs,
                actions, rewards]

    @partial(jit, static_argnums=(0,))
    def add_experience(self, cache, experience, actual):
        
        for idx, new in enumerate(experience):
            cache[idx] = jnp.concatenate((new[:,None], cache[idx][:,:-1]), axis=1)

        return cache, actual+1

    
    @partial(jit, static_argnums=(0,))
    def add_to_buffer(self, data, cache):
        
        cs = self.cache_size
        for idx, new in enumerate(cache):
            data[idx] = jnp.concatenate((new, data[idx][:,:-cs]), axis=1)
        
        cache = self.get_datastructure(self.cache_shape)

        return data, cache


    @partial(jit, static_argnums=(0,))
    def sample_from_buffer(self, key, data):
        bp, bs = self.bsp, self.bsn

        # select a number of protein pairs to draw examples from
        key, pkey = jrn.split(key, 2)
        uniq_pidxs = jrn.choice(
                pkey, self.pair_number, shape=(bp,), replace=True)

        # match selected pair idxs to cache and example idx
        pidxs = jnp.broadcast_to(uniq_pidxs[:, None], (bp, bs))
        pidxs = jnp.ravel(pidxs)

        # select for each example to add to the batch a cache example idx
        key, ekey = jrn.split(key, 2)
        eidxs = jrn.choice(
                ekey, self.cache_size, shape=(bp*bs,), replace=True)

        # select same random examples for different arrays in data
        # but with different idxs between different protein pairs
        batch = []
        for sub_data in data:
            examples = [sub_data[p][e] for p, e in zip(pidxs, eidxs)]
            batch.append(jnp.stack(examples))
 
        return batch, uniq_pidxs
