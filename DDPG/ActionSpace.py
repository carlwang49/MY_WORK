import numpy as np

class ActionSpace:
    """Define the action space for each agent in the environment"""
    def __init__(self, low, high, shape):
        self.low = low
        self.high = high
        self.shape = shape
 
    def sample(self, agents: list, agent_status: dict):
        sample = np.random.uniform(self.low, self.high)
        action = []

        for agent in agents:
            if agent_status[agent]:
                action.append(sample)
            else:
                action.append(0)    

        return action
    
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