import numpy as np
import torch

class TopBuffer:
    def __init__(self, capacity, obs_dim, device):
        self.capacity = capacity

        self.obs = np.zeros((capacity, obs_dim))
        self.action = np.zeros((capacity, 2))  # 储存动作概率分布
        self.reward = np.zeros(capacity)
        self.next_obs = np.zeros((capacity, obs_dim))
        self.done = np.zeros(capacity, dtype=bool)

        self._index = 0
        self._size = 0

        self.device = device

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self._index] = obs
        self.action[self._index] = action  # 假设 action 是一个概率分布
        self.reward[self._index] = reward
        self.next_obs[self._index] = next_obs
        self.done[self._index] = done

        self._index = (self._index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, batch_size):
        indices = np.random.choice(self._size, size=batch_size, replace=True)
        
        obs = self.obs[indices]
        action = self.action[indices]
        reward = self.reward[indices]
        next_obs = self.next_obs[indices]
        done = self.done[indices]

        obs = torch.from_numpy(obs).float().to(self.device)
        action = torch.from_numpy(action).float().to(self.device)
        reward = torch.from_numpy(reward).float().to(self.device)
        next_obs = torch.from_numpy(next_obs).float().to(self.device)
        done = torch.from_numpy(done).float().to(self.device)
        return obs, action, reward, next_obs, done

    def __len__(self):
        return self._size