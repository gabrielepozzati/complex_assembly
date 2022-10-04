import jax
import jax.numpy as jnp

class ReplayBuffer():
    def __init__(self, env, max_size, device = 'cpu'):
        self.env = env
        self.actual_size = 0
        self.max_size = max_size
        self.data = {pdb:[] for pdb in self.env.keys()}        

    def fill(self):
        while self.actual_size < self.max_size:
            actions = env.sample()
            new_state, reward, contacts = env.step(actions) 


            self.actual_size += 1

    def add(self):


    def sample(self, batch_size):

        return batch

