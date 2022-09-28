import jax
import jax.numpy as jnp
import psutil

class ReplayBuffer():
    def __init__(self, env, buffer_size = 1e5, device = 'cpu'):
        self.env = env
        self.buffer_size = buffer_size
       

    def fill(self):
        

    def add(self):


    def sample(self, batch_size):

        return batch

