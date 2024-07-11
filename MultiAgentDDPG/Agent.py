from copy import deepcopy
from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam
import numpy as np


class Agent:
    """Agent that can interact with environment from pettingzoo"""

    def __init__(self, obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = MLPNetwork(obs_dim, act_dim).to(self.device)
        # critic input all the observations and actions
        # if there are 3 agents for example, the input for critic is (obs1, obs2, obs3, act1, act2, act3)
        self.critic = MLPNetwork(global_obs_dim, 1).to(self.device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.target_actor = deepcopy(self.actor).to(self.device)
        self.target_critic = deepcopy(self.critic).to(self.device)
        self.action_values = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
        

    @staticmethod
    def gumbel_softmax(logits, tau=1.0, eps=1e-20):
        epsilon = torch.rand_like(logits)
        logits += -torch.log(-torch.log(epsilon + eps) + eps)
        probabilities = F.softmax(logits / tau, dim=-1)
        return probabilities

    def action(self, obs, agent_status, model_out=False):
        # this method is called in the following two cases:
        # a) interact with the environment
        # b) calculate action when update actor, where input(obs) is sampled from replay buffer with size:
        # torch.Size([batch_size, state_dim])
        obs = obs.to(self.device)
        logits = self.actor(obs)  # torch.Size([batch_size, action_size])
        
        if not agent_status:
            # Apply masking: setting logits of invalid actions to a large negative value
            logits[:, 0] = -1e10  # Assuming the invalid action corresponds to index 0
        
        # action = self.gumbel_softmax(logits)
        # action = F.gumbel_softmax(logits, hard=True)
        
        # Use tanh activation to ensure the action is in the range [-1, 1]
        action_continuous = torch.tanh(logits)
        # Convert action values to tensor
        action_values_tensor = torch.tensor(self.action_values, device=self.device, dtype=action_continuous.dtype)

        # Find the closest discrete action value
        action_indices = torch.argmin(torch.abs(action_continuous.unsqueeze(-1) - action_values_tensor.unsqueeze(0)), dim=-1)
        action = action_values_tensor[action_indices]
        
        # action = torch.tanh(logits)
        if model_out:
            return action, logits
        return action

    def target_action(self, obs, agent_status):
        """select action for all agents using their target actor network"""
        # when calculate target critic value in MADDPG,
        # we use target actor to get next action given next states,
        # which is sampled from replay buffer with size torch.Size([batch_size, state_dim])
        logits = self.target_actor(obs)  # torch.Size([batch_size, action_size])
        
        if not agent_status:
            # Apply masking: setting logits of invalid actions to a large negative value
            logits[:, 0] = -1e10  # Assuming the invalid action corresponds to index 0
            
        action_continuous = torch.tanh(logits)
        # Convert action values to tensor
        action_values_tensor = torch.tensor(self.action_values, device=self.device, dtype=action_continuous.dtype)

        # Find the closest discrete action value
        action_indices = torch.argmin(torch.abs(action_continuous.unsqueeze(-1) - action_values_tensor.unsqueeze(0)), dim=-1)
        action = action_values_tensor[action_indices]
        # action = self.gumbel_softmax(logits)
        # action = F.gumbel_softmax(logits, hard=True)
        # action = torch.tanh(logits)
        return action.squeeze(0).detach()

    def critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        """calculate the value of the critic network for a given state and action"""
        x = torch.cat(state_list + act_list, 1)
        return self.critic(x).squeeze(1)  # tensor with a given length

    def target_critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        """calculate the value of the target critic network for a given state and action"""
        x = torch.cat(state_list + act_list, 1)
        return self.target_critic(x).squeeze(1)  # tensor with a given length

    def update_actor(self, loss):
        """update actor network using the loss"""
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        """update critic network using the loss"""
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()


class MLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU()):
        super(MLPNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        ).apply(self.init)

    @staticmethod
    def init(m):
        """init parameter of the module"""
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)
