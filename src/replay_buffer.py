import jax
import jax.numpy as jnp

class ReplayBuffer():
    def __init__(self, env, max_size, key, device='cpu'):
        self.env = env
        self.key = key
        self.actual_size = 0
        self.max_size = max_size
        self.data = {pdb:Experience_List(max_size) for pdb in self.env.list}        
        jax.device_put(self.data, jax.devices(device)[0])

    def add(self, experiences):
        self.data = jax.tree_util.tree_map(lambda x, y: x.add(y), 
                self.data, experiences)

    def sample(self, batch_size):
        return jax.tree_util.tree_map(
                lambda x: jnp.random.choice(x.list, size=batch_size, replace=False)
                self.data)


class Experience:
    def __init__(self, 
            prev_state, next_state,
            action, reward, status):
        
        self.prev_state = prev_state
        self.next_state = next_state
        self.action = action
        self.reward = reward
        self.confidence = confidence

class Experience_List:
    def __init__(self, max_size):
        self.list = ()
        self.actual_size = 0
        self.max_sixe = max_size

    def add(self, experience):
        if self.actual_size == self.max_size:
            self.list[1:] += experience
        else:
            self.list += experience
            self.actual_size += 1
