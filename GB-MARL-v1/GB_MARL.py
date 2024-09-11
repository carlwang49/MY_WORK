import os
import pickle
import wandb
import numpy as np
import torch
import torch.nn.functional as F 
from TopAgent import TopAgent
from Buffer import Buffer
from TopBuffer import TopBuffer 
from utils import setup_logger
from ChargingAgent import ChargingAgent
from DischargingAgent import DischargingAgent


class GB_MARL:
    def __init__(self, dim_info, top_dim_info, capacity, batch_size, top_level_buffer_capacity, 
                 top_level_batch_size, actor_lr, critic_lr, res_dir):
        """dim_info: dict, key is agent_id, value is a tuple of obs_dim and act_dim"""
        
        # Top level agent
        top_level_obs_dim, top_level_act_dim = top_dim_info[0], top_dim_info[1] # observation and action dimension of the top level agent
        self.top_level_agent = TopAgent(top_level_obs_dim, top_level_act_dim, actor_lr, critic_lr) # initialize the top level agent
        self.top_level_buffer = TopBuffer(top_level_buffer_capacity, top_level_obs_dim, top_level_act_dim, 'cuda') # initialize the top level buffer
        
        self.charging_agent = {} # charging agents
        self.charging_buffers = {} # charging buffers
        self.discharging_agent = {} # discharging agents
        self.discharging_buffers = {} # discharging buffers
        
        global_obs_act_dim = sum(sum(val) for val in dim_info.values()) # global observation and action dimension
        for agent_id, (obs_dim, act_dim) in dim_info.items():
            self.charging_agent[agent_id] = ChargingAgent(obs_dim, act_dim, global_obs_act_dim, actor_lr, critic_lr)
            self.charging_buffers[agent_id] = Buffer(capacity, obs_dim, act_dim, 'cuda')
        
        for agent_id, (obs_dim, act_dim) in dim_info.items():
            self.discharging_agent[agent_id] = DischargingAgent(obs_dim, act_dim, global_obs_act_dim, actor_lr, critic_lr)
            self.discharging_buffers[agent_id] = Buffer(capacity, obs_dim, act_dim, 'cuda')
        
        self.dim_info = dim_info # observation and action dimension of all agents
        self.batch_size = batch_size # batch size
        self.top_level_batch_size = top_level_batch_size # top level batch size
        self.res_dir = res_dir # result directory
        # self.logger = setup_logger(os.path.join(res_dir, 'GB_MARL.log')) # initialize the logger


    def add(self, obs, global_observation, action, top_level_action, reward, global_reward, next_obs, next_global_observation, done):
        """Add the transition to the buffer"""
        
        if top_level_action == 0: # charging
            # add the transition to the charging buffer
            for agent_id in obs.keys():
                o = obs[agent_id]
                a = action[agent_id]
                r = reward[agent_id]
                next_o = next_obs[agent_id]
                d = done[agent_id]
                self.charging_buffers[agent_id].add(o, a, r, next_o, d) # add the transition to the buffer
        else: # discharging
            # add the transition to the discharging buffer
            for agent_id in obs.keys():
                o = obs[agent_id]
                a = action[agent_id]
                r = reward[agent_id]
                next_o = next_obs[agent_id]
                d = done[agent_id]
                self.discharging_buffers[agent_id].add(o, a, r, next_o, d) # add the transition to the buffer
        
        top_level_obs = global_observation 
        top_level_next_obs = next_global_observation
        top_level_reward = global_reward
        top_level_done = any(done.values())
        
        # add the transition to the top level buffer
        self.top_level_buffer.add(top_level_obs, top_level_action, top_level_reward, top_level_next_obs, top_level_done)


    def sample(self, batch_size, agents_status, top_level_action):
        """Sample transitions from the buffer"""
        obs, act, reward, next_obs, done, next_act = {}, {}, {}, {}, {}, {}
        
        if top_level_action == 0: # charging
            total_num = len(self.charging_buffers['agent_0'])
            indices = np.random.choice(total_num, size=batch_size, replace=True)
            
            for agent_id, buffer in self.charging_buffers.items():
                o, a, r, n_o, d = buffer.sample(indices) # sample transitions from the buffer
                obs[agent_id] = o
                act[agent_id] = a
                reward[agent_id] = r
                next_obs[agent_id] = n_o
                done[agent_id] = d
                next_act[agent_id] = \
                    self.charging_agent[agent_id].target_action(n_o, agents_status[agent_id]) # get the next action using the target actor network
        else: # discharging
            total_num = len(self.discharging_buffers['agent_0'])
            indices = np.random.choice(total_num, size=batch_size, replace=True)
    
            for agent_id, buffer in self.discharging_buffers.items():
                o, a, r, n_o, d = buffer.sample(indices) # sample transitions from the buffer
                obs[agent_id] = o
                act[agent_id] = a
                reward[agent_id] = r
                next_obs[agent_id] = n_o
                done[agent_id] = d
                next_act[agent_id] = \
                    self.discharging_agent[agent_id].target_action(n_o, agents_status[agent_id]) # get the next action using the target actor network    
        
        return obs, act, reward, next_obs, done, next_act
    
    def top_level_sample(self, batch_size):
        """Sample transitions from the top level buffer"""
        top_level_obs, top_level_act, top_level_reward, top_level_next_obs, top_level_done = self.top_level_buffer.sample(batch_size)
        top_level_next_act = self.top_level_agent.target_action(top_level_next_obs)
        return top_level_obs, top_level_act, top_level_reward, top_level_next_obs, top_level_done, top_level_next_act


    def select_action(self, obs, global_observation, agents_status):
        """Select action using GB_MARL"""
        
        top_level_obs = global_observation
        top_level_action = self.top_level_agent.action(torch.from_numpy(top_level_obs).unsqueeze(0).float()) # select top level action
        top_level_action = int(top_level_action.item()) # get the top level action

        actions = {}
        if top_level_action == 0: # charging
            for agent, o in obs.items():
                o = torch.from_numpy(o).unsqueeze(0).float()
                a = self.charging_agent[agent].action(o, agents_status[agent])
                actions[agent] = a.squeeze(0).item()
                # self.logger.info(f'{agent} action: {actions[agent]}')
        else: # discharging
            for agent, o in obs.items():
                o = torch.from_numpy(o).unsqueeze(0).float()
                a = self.discharging_agent[agent].action(o, agents_status[agent])
                actions[agent] = a.squeeze(0).item()
                # self.logger.info(f'{agent} action: {actions[agent]}')
                
        return actions, top_level_action


    def learn(self, batch_size, top_level_batch_size, gamma, agents_status, step):
        """Learn from the replay buffer"""
        
        for agent_id, agent in self.charging_agent.items(): # learn from the charging buffer
            obs, act, reward, next_obs, done, next_act = self.sample(batch_size, agents_status, top_level_action=0)
            critic_value = agent.critic_value(list(obs.values()), list(act.values())) # calculate the value of the critic network
            next_target_critic_value = agent.target_critic_value(list(next_obs.values()), list(next_act.values())) # calculate the value of the target critic network
            target_value = reward[agent_id] + gamma * next_target_critic_value * (1 - done[agent_id]) # calculate the target value
            critic_loss = F.mse_loss(critic_value, target_value.detach(), reduction='mean') # calculate the critic loss
            agent.update_critic(critic_loss) # update the critic network
            
            advantage = target_value - critic_value.detach()
            action, logits = agent.action(obs[agent_id], agents_status[agent_id], model_out=True) # select action
            act[agent_id] = action # update the action
            actor_loss = -(advantage * agent.critic_value(list(obs.values()), list(act.values()))).mean() # calculate the actor loss using advantage
            actor_loss_pse = torch.pow(logits, 2).mean() # calculate the actor loss pse
            agent.update_actor(actor_loss + 1e-3 * actor_loss_pse) # update the actor network
            # self.logger.info(f'{agent_id}: critic loss: {critic_loss.item()}, actor loss: {actor_loss.item()}')
            
            wandb.log({
                "charging agent critic loss:": critic_loss.item(),
                "charging agent actor loss": actor_loss.item()
            }, step=step)
            
        for agent_id, agent in self.discharging_agent.items(): # learn from the discharging buffer
            obs, act, reward, next_obs, done, next_act = self.sample(batch_size, agents_status, top_level_action=1)
            critic_value = agent.critic_value(list(obs.values()), list(act.values()))
            next_target_critic_value = agent.target_critic_value(list(next_obs.values()), list(next_act.values()))
            target_value = reward[agent_id] + gamma * next_target_critic_value * (1 - done[agent_id])
            critic_loss = F.mse_loss(critic_value, target_value.detach(), reduction='mean')
            agent.update_critic(critic_loss)
            
            advantage = target_value - critic_value.detach()
            action, logits = agent.action(obs[agent_id], agents_status[agent_id], model_out=True)
            act[agent_id] = action
            actor_loss = -(advantage * agent.critic_value(list(obs.values()), list(act.values()))).mean() # calculate the actor loss using advantage
            actor_loss_pse = torch.pow(logits, 2).mean()
            agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)
            # self.logger.info(f'{agent_id}: critic loss: {critic_loss.item()}, actor loss: {actor_loss.item()}')
            wandb.log({
                "discharging agent critic loss:": critic_loss.item(),
                "discharging agent actor loss": actor_loss.item()
            }, step=step)
        
        # top level agent
        top_level_obs, top_level_act, top_level_reward, top_level_next_obs, top_level_done, top_level_next_act = self.top_level_sample(top_level_batch_size) # sample transitions from the top level buffer
        top_level_critic_value = self.top_level_agent.critic_value(top_level_obs, top_level_act) # calculate the value of the critic network
        next_top_target_critic_value = self.top_level_agent.target_critic_value(top_level_next_obs, top_level_next_act) # calculate the value of the target critic network
        top_target_value = top_level_reward + gamma * next_top_target_critic_value * (1 - top_level_done) # calculate the target value
        top_critic_loss = F.mse_loss(top_level_critic_value, top_target_value.detach(), reduction='mean') # calculate the critic loss
        self.top_level_agent.update_critic(top_critic_loss) # update the critic network
        
        top_advantage = top_target_value - top_level_critic_value.detach()
        top_action, top_logits = self.top_level_agent.action(top_level_obs, model_out=True) # select action
        top_level_actor_loss = -(top_advantage * self.top_level_agent.critic_value(top_level_obs, top_level_act)).mean() # calculate the actor loss using advantage
        top_level_actor_loss_pse = torch.pow(top_logits, 2).mean() # calculate the actor loss pse
        self.top_level_agent.update_actor(top_level_actor_loss + 1e-3 * top_level_actor_loss_pse) # update the actor network
        # self.logger.info(f'Top Level Agent: critic loss: {top_critic_loss.item()}, actor loss: {top_level_actor_loss.item()}') 
        
        wandb.log({
            "top agent critic loss:": top_critic_loss.item(),
            "top agent actor loss": top_level_actor_loss.item()
        }, step=step)
        
        
    def update_target(self, tau):
        """Update the target network"""
        def soft_update(from_network, to_network):
            """Soft update the target network"""
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                # Update the target network parameters using the soft update rule
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        # update the target network for all agents
        for agent in self.charging_agent.values():
            soft_update(agent.actor, agent.target_actor)
            soft_update(agent.critic, agent.target_critic)
            
        for agent in self.discharging_agent.values():
            soft_update(agent.actor, agent.target_actor)
            soft_update(agent.critic, agent.target_critic)
            
        # update the target network for the top level agent
        soft_update(self.top_level_agent.actor, self.top_level_agent.target_actor)  
        soft_update(self.top_level_agent.critic, self.top_level_agent.target_critic)

    def save(self, reward):
        model_state = {
            'charging_agent_actor': {name: agent.actor.state_dict() for name, agent in self.charging_agent.items()},
            'charging_agent_critic': {name: agent.critic.state_dict() for name, agent in self.charging_agent.items()},
            'discharging_agent_actor': {name: agent.actor.state_dict() for name, agent in self.discharging_agent.items()},
            'discharging_agent_critic': {name: agent.critic.state_dict() for name, agent in self.discharging_agent.items()},
            'top_level_agent_actor': self.top_level_agent.actor.state_dict(),
            'top_level_agent_critic': self.top_level_agent.critic.state_dict()
        }
        torch.save(model_state, os.path.join(self.res_dir, 'model.pt'))
        
        with open(os.path.join(self.res_dir, 'rewards.pkl'), 'wb') as f:
            pickle.dump({'rewards': reward}, f)


    @classmethod
    def load(cls, dim_info, top_dim_info, actor_lr, critic_lr, res_dir, file):
        """Load the model"""
        # Initialize the instance with required parameters
        instance = cls(dim_info, top_dim_info, capacity=0, batch_size=0, 
                       top_level_buffer_capacity=0, top_level_batch_size=0, 
                       actor_lr=actor_lr, critic_lr=critic_lr, res_dir=res_dir)
        data = torch.load(file)
        
        for agent_id, agent in instance.charging_agent.items():
            agent.actor.load_state_dict(data['charging_agent_actor'][agent_id])
            agent.critic.load_state_dict(data['charging_agent_critic'][agent_id])
        
        for agent_id, agent in instance.discharging_agent.items():
            agent.actor.load_state_dict(data['discharging_agent_actor'][agent_id])
            agent.critic.load_state_dict(data['discharging_agent_critic'][agent_id])
        
        instance.top_level_agent.actor.load_state_dict(data['top_level_agent_actor'])
        instance.top_level_agent.critic.load_state_dict(data['top_level_agent_critic'])
        
        return instance  # return the instance