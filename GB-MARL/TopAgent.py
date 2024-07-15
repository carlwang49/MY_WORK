import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from copy import deepcopy

class TopAgent:
    def __init__(self, obs_dim, act_dim, actor_lr):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = MLPNetwork(obs_dim, act_dim).to(self.device)
        self.critic = MLPNetwork(obs_dim + act_dim, 1).to(self.device)  # critic 输入是状态和动作
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=actor_lr)
        self.target_actor = deepcopy(self.actor).to(self.device)
        self.target_critic = deepcopy(self.critic).to(self.device)
        
    def action(self, obs, model_out=False):
        obs = obs.to(self.device)
        logits = self.actor(obs)
        action = F.softmax(logits, dim=-1)  # 返回概率分布
        if model_out:
            return action, logits
        return action
    
    def target_action(self, obs):
        logits = self.target_actor(obs)
        action = F.softmax(logits, dim=-1)
        return action
    
    def critic_value(self, obs, act):
        inputs = torch.cat((obs, act), dim=1)
        return self.critic(inputs).squeeze(1)
    
    def target_critic_value(self, obs, act):
        inputs = torch.cat((obs, act), dim=1)
        return self.target_critic(inputs).squeeze(1)

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
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super(MLPNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        ).apply(self.init)

    @staticmethod
    def init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)