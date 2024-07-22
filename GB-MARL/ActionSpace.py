import numpy as np

class ActionSpace:
    """Define the action space for each agent in the environment"""
    def __init__(self, low, high, shape):
        self.low = low
        self.high = high
        self.shape = shape
 
    def sample(self):
        sample = np.random.uniform(self.low, self.high, self.shape)
        return round(float(sample), 4)
    
    def size(self):
        return self.shape[0]
    
class DiscreteActionSpace:
    """Define the action space for each agent in the environment"""
    
    def __init__(self, action_values):
        self.action_values = np.array(action_values)
        self.n = 1  # Define the number of actions

    def sample(self):
        sample = np.random.choice(self.action_values)
        return sample
    
    def size(self):
        return self.n
    

class TopLevelActionSpace:
    """Define the action space for the top level agent in the environment"""
    
    def __init__(self, action_values):
        self.action_values = np.array(action_values)
        self.n = 1 # Define the number of actions

    def sample(self):
        sample = np.random.choice(self.action_values)
        return sample
    
    def size(self):
        return self.n