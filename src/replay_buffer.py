import sys
import jax
import jax.numpy as jnp
import jax.random as jrn
from jax.tree_util import tree_map
from functools import partial
from jax import jit

class ReplayBuffer():
    def __init__(self, config, plist):
        
        self.pair_list = plist
        self.pair_number = len(plist)
        self.enum = config['edge_number']
        self.pad = config['padding']

        self.buff_actual = 0
        self.cache_actual = 0

        self.buff_size = config['buffer_size']

        c_num = self.pair_number
        i_num = c_num*self.pad*self.enum*2*2*2
        e_num = c_num*self.pad*self.enum*2*15*2*4
        m_num = c_num*self.pad*2*2*2
        a_num = c_num*3*self.pad*4
        r_num = c_num*4

        tot_ex = i_num+e_num+m_num+a_num+r_num
        self.cache_size = int(jnp.floor((config['cache_gb']*1e9)/tot_ex))
        self.cache_size = min(self.cache_size, self.buff_size)

        self.bsn = config['batch_size_num']
        self.bsp = config['batch_size_pair']

        buff_shape = (self.pair_number, self.buff_size)
        self.cache_shape = (self.pair_number, self.cache_size)
    
        self.buff = self.get_datastructure(buff_shape)
        self.cache = self.get_datastructure(self.cache_shape)
        self.cache = jax.device_put(self.cache, device=jax.devices('gpu')[0])

        tot = tot_ex * self.buff_size
        print (f'Single example size {round((tot_ex/1e9),3)}GB')
        print (f'Number of examples in cache:', self.cache_size)
        print (f'Buffer size: {round((tot/1e9),3)}GB')


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
    def add_to_cache(self, cache, experience, actual):
        
        for idx, new in enumerate(experience):
            cache[idx] = jnp.concatenate((new[:,None], cache[idx][:,:-1]), axis=1)

        return cache, actual+1

    
    @partial(jit, static_argnums=(0,))
    def add_to_buffer(self, data, cache, actual):
        
        c_s = self.cache_size
        for idx, new in enumerate(cache):
            data[idx] = jnp.concatenate((cache[idx], data[idx][:,:-c_s]), axis=1)

        cache = self.get_datastructure(self.cache_shape)

        return data, cache, actual+c_s


    @partial(jit, static_argnums=(0,))
    def sample_from_buffer(self, key, data, actual):

        def _sample(ex_data, nkey):
            ex_idxs = jrn.choice(nkey, actual, shape=(self.bsn,), replace=False)
            return ex_data[ex_idxs]

        # select a number of protein pairs to draw examples from
        key, pkey = jrn.split(key, 2)
        pair_idxs = jrn.choice(pkey, self.pair_num, shape=(self.bsp,), replace=False)
        
        # create an array with a different key for each pair
        nkeys = jrn.split(key, self.bsp)

        # select same random examples for different arrays in data 
        # but with different idxs between different protein pairs
        batch = []
        for pair_ex_data in data:
            batch.append(vmap(_sample)(pair_ex_data[pair_idxs], nkeys))
        
        return batch
