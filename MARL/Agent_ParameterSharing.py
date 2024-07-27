from copy import deepcopy
from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam


class Agent:
    """Agent that can interact with environment from pettingzoo"""

    def __init__(self, actor, critic, actor_optimizer, critic_optimizer):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.target_actor = deepcopy(actor)
        self.target_critic = deepcopy(critic)
        

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
        action = torch.tanh(logits)
        if model_out:
            return action, logits
        return action

    def target_action(self, obs, agent_status):
        # when calculate target critic value in MADDPG,
        # we use target actor to get next action given next states,
        # which is sampled from replay buffer with size torch.Size([batch_size, state_dim])
        logits = self.target_actor(obs)  # torch.Size([batch_size, action_size])
        
        if not agent_status:
            # Apply masking: setting logits of invalid actions to a large negative value
            logits[:, 0] = -1e10  # Assuming the invalid action corresponds to index 0
            
        # action = self.gumbel_softmax(logits)
        # action = F.gumbel_softmax(logits, hard=True)
        action = torch.tanh(logits)
        return action.squeeze(0).detach()

    def critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.critic(x).squeeze(1)  # tensor with a given length

    def target_critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.target_critic(x).squeeze(1)  # tensor with a given length

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
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
