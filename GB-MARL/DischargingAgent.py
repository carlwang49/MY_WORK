from copy import deepcopy
from typing import List
import torch
from torch import nn, Tensor
from torch.optim import Adam


class DischargingAgent:
    """Agent that can interact with environment from pettingzoo"""

    def __init__(self, obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = MLPNetwork(obs_dim, act_dim).to(self.device)
        self.critic = MLPNetwork(global_obs_dim, 1).to(self.device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.target_actor = deepcopy(self.actor).to(self.device)
        self.target_critic = deepcopy(self.critic).to(self.device)

    def action(self, obs, agent_status, model_out=False):
        """select action for all agents using their actor network"""
        obs = obs.to(self.device)
        logits = self.actor(obs)  # torch.Size([batch_size, action_size])
        
        if not agent_status:
            # Apply masking: setting logits of invalid actions to a large negative value
            logits[:, 0] = -1e10  # Assuming the invalid action corresponds to index 0
        
        action = logits
        if model_out:
            return action, logits
        return action

    def target_action(self, obs, agent_status):
        """select action for all agents using their target actor network"""
        logits = self.target_actor(obs)  # torch.Size([batch_size, action_size])
        
        if not agent_status:
            # Apply masking: setting logits of invalid actions to a large negative value
            logits[:, 0] = -1e10  # Assuming the invalid action corresponds to index 0
            
        action = logits
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


class ScaledSigmoid(nn.Module):
    def __init__(self, scale=0.5, shift=0):
        super(ScaledSigmoid, self).__init__()
        self.scale = scale
        self.shift = shift

    def forward(self, x):
        return torch.sigmoid(x) * self.scale + self.shift


class MLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU(), output_scale=0.5, output_shift=0):
        super(MLPNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GroupNorm(1, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            nn.GroupNorm(1, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
            ScaledSigmoid(scale=output_scale, shift=output_shift) 
        ).apply(self.init)

    @staticmethod
    def init(m):
        """init parameter of the module"""
        gain = nn.init.calculate_gain('relu') # gain for the ReLU activation function
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)
