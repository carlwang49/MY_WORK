import numpy as np
import torch

class TopBuffer:
    def __init__(self, capacity, top_level_obs_dim, top_level_act_dim, device):
        self.capacity = capacity # the capacity of the buffer
        self.obs = np.zeros((capacity, top_level_obs_dim)) # store the observation
        self.action = np.zeros((capacity, top_level_act_dim))  # store the probability of each action
        self.reward = np.zeros(capacity) # store the reward
        self.next_obs = np.zeros((capacity, top_level_obs_dim)) # store the next observation
        self.done = np.zeros(capacity, dtype=bool) # store the done
        self._index = 0 # the index of the buffer
        self._size = 0 # the size of the buffer
        self.device = device # the device to store the data

    def add(self, obs, action, reward, next_obs, done):
        """ add an experience to the memory """
        self.obs[self._index] = obs # store the observation
        self.action[self._index] = action  # store the probability of each action
        self.reward[self._index] = reward # store the reward
        self.next_obs[self._index] = next_obs # store the next observation
        self.done[self._index] = done # store the done

        # update the index
        self._index = (self._index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, batch_size):
        """sample a batch of experiences from the buffer"""
        indices = np.random.choice(self._size, size=batch_size, replace=True) # sample the indices
        
        obs = self.obs[indices] # sample the observation
        action = self.action[indices] # sample the probability of each action
        reward = self.reward[indices] # sample the reward
        next_obs = self.next_obs[indices] # sample the next observation
        done = self.done[indices] # sample the done

        # convert the data to tensor
        obs = torch.from_numpy(obs).float().to(self.device) 
        action = torch.from_numpy(action).float().to(self.device) 
        reward = torch.from_numpy(reward).float().to(self.device)
        next_obs = torch.from_numpy(next_obs).float().to(self.device)
        done = torch.from_numpy(done).float().to(self.device)
        return obs, action, reward, next_obs, done

    def __len__(self):
        return self._size