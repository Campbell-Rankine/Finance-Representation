from collections import deque
import random

class ReplayBuffer(object):
    """
    Continuous DDPG replay memory buffer for storing state tuples to sample
    """
    def __init__(self, buffer_size):
        self.size = buffer_size
        self.storage = deque()
        self.num_exp = 0

    def add(self, state, action, reward, new_state, done):
        if self.available():
            self.storage.append((state, action, reward, new_state, done))
            self.num_exp += 1
        else:
            self.storage.popleft()
            self.storage.append((state, action, reward, new_state, done))
    
    def sample(self, sample_size):
        if self.num_experiences < sample_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, sample_size)
    
    def reset(self):
        self.storage = deque()
        self.num_exp = 0

    
    ### - ATTRIBUTE FUNCTIONS - ###

    def size(self):
        return self.size

    def available(self):
        """
        Boolean function, finds if memory has space for an additional state
        """
        return (self.size - (self.num_exp + 1)) <= 0
    