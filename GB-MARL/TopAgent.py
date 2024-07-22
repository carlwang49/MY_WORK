import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from copy import deepcopy
import numpy as np

class TopAgent:
    def __init__(self, obs_dim, act_dim, actor_lr, critic_lr, epsilon=0.1, sigma=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = MLPNetwork(obs_dim, act_dim).to(self.device) # actor network
        self.critic = MLPNetwork(obs_dim + act_dim, 1).to(self.device) # critic network
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr) # actor optimizer
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr) # critic optimizer
        self.target_actor = deepcopy(self.actor).to(self.device) # target actor
        self.target_critic = deepcopy(self.critic).to(self.device) # target critic
        self.epsilon = epsilon
        self.sigma = sigma
        
    # def action(self, obs, model_out=False):
    #     """Return the action of the agent"""
    #     obs = obs.to(self.device)
    #     logits = self.actor(obs)
    #     action = (logits > 0.5).float() # return the action
    #     noise = torch.normal(0, self.sigma, size=action.size()).to(self.device)
    #     action = action + noise
    #     action = torch.clamp(action, 0, 1)  # 将动作限制在 0 到 1 之间
    #     if model_out:
    #         # return the action and the logits
    #         return action, logits
    #     return action
    
    def action(self, obs, model_out=False):
        """Return the action of the agent"""
        obs = obs.to(self.device)
        logits = self.actor(obs)
        noise = torch.normal(0, self.sigma, size=logits.size()).to(self.device)
        logits = logits + noise  # 在 logits 上加噪声
        action = (logits > 0.5).float()  # 将 logits 转换为 0 或 1
        if model_out:
            # return the action and the logits
            return action, logits
        return action
    
    
    def target_action(self, obs):
        """Return the action of the target actor"""
        logits = self.target_actor(obs)
        action = (logits > 0.5).float()
        return action
    
    def critic_value(self, obs, act):
        """Return the value of the critic network"""
        inputs = torch.cat((obs, act), dim=1) # concatenate the observation and action
        return self.critic(inputs).squeeze(1) # return the value of the critic network
    
    def target_critic_value(self, obs, act):
        """Return the value of the target critic network"""
        inputs = torch.cat((obs, act), dim=1) # concatenate the observation and action
        return self.target_critic(inputs).squeeze(1) # return the value of the target critic network

    def update_actor(self, loss):
        """Update the actor network"""
        self.actor_optimizer.zero_grad() # zero the gradients of the optimizer
        loss.backward() # backpropagate the loss
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5) # clip the gradients is becase of the exploding gradient problem
        self.actor_optimizer.step() # update the parameters of the actor network
        
    def update_critic(self, loss):
        """Update the critic network"""
        self.critic_optimizer.zero_grad() # zero the gradients of the optimizer
        loss.backward() # backpropagate the loss
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5) # clip the gradients is becase of the exploding gradient problem
        self.critic_optimizer.step() # update the parameters of the critic network



class MLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        # input size: in_dim, output size: out_dim
        # hidden layer size: hidden_dim
        super(MLPNetwork, self).__init__() # inherit the properties of the parent class

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid()
        ).apply(self.init)

    @staticmethod
    def init(m):
        # initialize the parameters of the network
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain) # initialize the weight using xavier uniform
            m.bias.data.fill_(0.01) # initialize the bias to 0.01

    def forward(self, x):
        # forward pass of the network
        return self.net(x) # return the output of the network