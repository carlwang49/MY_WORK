import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from Agent import Agent
from TopAgent import TopAgent
from Buffer import Buffer
from TopBuffer import TopBuffer    

def setup_logger(filename):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(filename, mode='w')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s--%(levelname)s--%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger

class MADDPG:
    def __init__(self, dim_info, capacity, batch_size, actor_lr, critic_lr, res_dir):
        
        top_level_obs_dim = sum(val[0] for val in dim_info.values())
        self.top_level_agent = TopAgent(top_level_obs_dim, 2, actor_lr)
        self.top_level_buffer = TopBuffer(capacity, top_level_obs_dim, 'cuda')
        
        global_obs_act_dim = sum(sum(val) for val in dim_info.values())
        self.agents = {}
        self.buffers = {}
        
        for agent_id, (obs_dim, act_dim) in dim_info.items():
            self.agents[agent_id] = Agent(obs_dim, act_dim, global_obs_act_dim, actor_lr, critic_lr)
            self.buffers[agent_id] = Buffer(capacity, obs_dim, act_dim, 'cuda')
        
        self.dim_info = dim_info
        self.batch_size = batch_size
        self.res_dir = res_dir
        self.logger = setup_logger(os.path.join(res_dir, 'maddpg.log'))

    def add(self, obs, action, reward, next_obs, done):
        for agent_id in obs.keys():
            o = obs[agent_id]
            a = action[agent_id]
            if isinstance(a, int):
                a = np.eye(self.dim_info[agent_id][1])[a]
            r = reward[agent_id]
            next_o = next_obs[agent_id]
            d = done[agent_id]
            self.buffers[agent_id].add(o, a, r, next_o, d)
        
        top_level_obs = np.concatenate([obs[agent_id] for agent_id in obs.keys()])
        top_level_next_obs = np.concatenate([next_obs[agent_id] for agent_id in obs.keys()])
        top_level_reward = sum(reward.values()) / len(reward)
        top_level_done = any(done.values())
        
        top_level_action = sum(action.values()) / len(action)
        self.top_level_buffer.add(top_level_obs, top_level_action, top_level_reward, top_level_next_obs, top_level_done)

    def sample(self, batch_size, agents_status):
        total_num = len(self.buffers['agent_0'])
        indices = np.random.choice(total_num, size=batch_size, replace=True)
 
        obs, act, reward, next_obs, done, next_act = {}, {}, {}, {}, {}, {}
        for agent_id, buffer in self.buffers.items():
            o, a, r, n_o, d = buffer.sample(indices)
            obs[agent_id] = o
            act[agent_id] = a
            reward[agent_id] = r
            next_obs[agent_id] = n_o
            done[agent_id] = d
            next_act[agent_id] = self.agents[agent_id].target_action(n_o, agents_status[agent_id])

        return obs, act, reward, next_obs, done, next_act

    def select_action(self, obs, agents_status):
        top_level_obs = np.concatenate([obs[agent_id] for agent_id in obs.keys()])
        top_level_action = self.top_level_agent.action(torch.from_numpy(top_level_obs).unsqueeze(0).float())
        top_level_action = torch.argmax(top_level_action, dim=1).item()
        
        actions = {}
        for agent, o in obs.items():
            o = torch.from_numpy(o).unsqueeze(0).float()
            a = self.agents[agent].action(o, agents_status[agent])
            actions[agent] = a.squeeze(0).item()
            self.logger.info(f'{agent} action: {actions[agent]}')
        return actions, top_level_action


    def learn(self, batch_size, gamma, agents_status):
        for agent_id, agent in self.agents.items():
            obs, act, reward, next_obs, done, next_act = self.sample(batch_size, agents_status)
            critic_value = agent.critic_value(list(obs.values()), list(act.values()))
            next_target_critic_value = agent.target_critic_value(list(next_obs.values()), list(next_act.values()))
            target_value = reward[agent_id] + gamma * next_target_critic_value * (1 - done[agent_id])
            critic_loss = F.mse_loss(critic_value, target_value.detach(), reduction='mean')
            agent.update_critic(critic_loss)

            action, logits = agent.action(obs[agent_id], agents_status[agent_id], model_out=True)
            act[agent_id] = action
            actor_loss = -agent.critic_value(list(obs.values()), list(act.values())).mean()
            actor_loss_pse = torch.pow(logits, 2).mean()
            agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)
            self.logger.info(f'{agent_id}: critic loss: {critic_loss.item()}, actor loss: {actor_loss.item()}')
        
        # 上层智能体学习
        top_level_obs, top_level_act, top_level_reward, top_level_next_obs, top_level_done = self.top_level_buffer.sample(batch_size)
        top_level_critic_value = self.top_level_agent.critic_value(top_level_obs, top_level_act)
        next_top_target_critic_value = self.top_level_agent.target_critic_value(top_level_next_obs, top_level_act)
        top_target_value = top_level_reward + gamma * next_top_target_critic_value * (1 - top_level_done)

        top_critic_loss = F.mse_loss(top_level_critic_value, top_target_value.detach(), reduction='mean')
        self.top_level_agent.update_critic(top_critic_loss)
        
        top_action, top_logits = self.top_level_agent.action(top_level_obs, model_out=True)
        top_level_actor_loss = -self.top_level_agent.critic_value(top_level_obs, top_action).mean()
        top_level_actor_loss_pse = torch.pow(top_logits, 2).mean()
        self.top_level_agent.update_actor(top_level_actor_loss + 1e-3 * top_level_actor_loss_pse)
        self.logger.info(f'Top Level Agent: critic loss: {top_critic_loss.item()}, actor loss: {top_level_actor_loss.item()}')
        
    def update_target(self, tau):
        def soft_update(from_network, to_network):
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        for agent in self.agents.values():
            soft_update(agent.actor, agent.target_actor)
            soft_update(agent.critic, agent.target_critic)

    def save(self, reward):
        torch.save(
            {name: agent.actor.state_dict() for name, agent in self.agents.items()},
            os.path.join(self.res_dir, 'model.pt')
        )
        with open(os.path.join(self.res_dir, 'rewards.pkl'), 'wb') as f:
            pickle.dump({'rewards': reward}, f)

    @classmethod
    def load(cls, dim_info, file):
        instance = cls(dim_info, 0, 0, 0, 0, os.path.dirname(file))
        data = torch.load(file)
        for agent_id, agent in instance.agents.items():
            agent.actor.load_state_dict(data[agent_id])
        return instance