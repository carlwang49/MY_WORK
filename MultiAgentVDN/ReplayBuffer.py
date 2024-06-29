import numpy as np
import collections
import torch

class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample_chunk(self, batch_size, chunk_size):
        start_idx = np.random.randint(0, len(self.buffer) - chunk_size, batch_size)
        # s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        s_lst, a_lst, r_lst, s_prime_lst, connected_lst = [], [], [], [], []

        for idx in start_idx:
            for chunk_step in range(idx, idx + chunk_size):
                s, a, r, s_prime, connected = self.buffer[chunk_step]
                s_lst.append(s)
                a_lst.append(a)
                r_lst.append(r)
                s_prime_lst.append(s_prime)
                connected_lst.append(connected)
                # done_lst.append(done)

        n_agents, obs_size = len(s_lst[0]), len(s_lst[0][0])
        return  torch.tensor(s_lst, dtype=torch.float).view(batch_size, chunk_size, n_agents, obs_size), \
                torch.tensor(a_lst, dtype=torch.float).view(batch_size, chunk_size, n_agents), \
                torch.tensor(r_lst, dtype=torch.float).view(batch_size, chunk_size, n_agents), \
                torch.tensor(s_prime_lst, dtype=torch.float).view(batch_size, chunk_size, n_agents, obs_size), \
                torch.tensor(connected_lst, dtype=torch.float).view(batch_size, chunk_size, 1)
            #    torch.tensor(done_lst, dtype=torch.float).view(batch_size, chunk_size, 1)

    def size(self):
        return len(self.buffer)