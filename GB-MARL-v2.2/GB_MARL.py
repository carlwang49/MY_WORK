import wandb
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F 
from TopAgent import TopAgent
from Buffer import Buffer
from TopBuffer import TopBuffer 
from utils import setup_logger
from Agent import Agent


class GB_MARL:
    def __init__(self, dim_info, top_dim_info, capacity, batch_size, top_level_buffer_capacity, 
                 top_level_batch_size, actor_lr, critic_lr, res_dir):
        """dim_info: dict, key is agent_id, value is a tuple of obs_dim and act_dim"""
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Top level agent
        top_level_obs_dim, top_level_act_dim = top_dim_info[0], top_dim_info[1] # observation and action dimension of the top level agent
        self.top_level_agent = TopAgent(top_level_obs_dim, top_level_act_dim, actor_lr, critic_lr) # initialize the top level agent
        self.top_level_buffer = TopBuffer(top_level_buffer_capacity, top_level_obs_dim, top_level_act_dim, 'cuda') # initialize the top level buffer
        
        # create Agent(actor-critic) and replay buffer for each agent
        self.agents = {}
        self.buffers = {}
        global_obs_act_dim = sum(sum(val) for val in dim_info.values()) # global observation and action dimension
        # dim_info is a dict with agent_id as key and (obs_dim, act_dim) as value
        for agent_id, (obs_dim, act_dim) in dim_info.items():
            self.agents[agent_id] = Agent(obs_dim, act_dim, global_obs_act_dim, actor_lr, critic_lr)
            self.buffers[agent_id] = Buffer(capacity, obs_dim, act_dim, 'cuda')
        
        self.dim_info = dim_info # observation and action dimension of all agents
        self.batch_size = batch_size # batch size
        self.top_level_batch_size = top_level_batch_size # top level batch size
        self.res_dir = res_dir # result directory
        # self.logger = setup_logger(os.path.join(res_dir, 'GB_MARL.log')) # initialize the logger


    def add(self, obs, global_observation, action, top_level_action, reward, global_reward, 
            next_obs, next_global_observation, current_hour, done):
        """Add the transition to the buffer"""
        
        # NOTE that the experience is a dict with agent name as its key
        for agent_id in obs.keys():
            o = obs[agent_id]
            a = action[agent_id]
            r = reward[agent_id]
            next_o = next_obs[agent_id]
            d = done[agent_id]
            self.buffers[agent_id].add(o, a, r, current_hour, next_o, d) # add experience to the buffer of each agent
        
        top_level_obs = global_observation 
        top_level_next_obs = next_global_observation
        top_level_reward = global_reward
        top_level_done = any(done.values())
        
        # add the transition to the top level buffer
        self.top_level_buffer.add(top_level_obs, top_level_action, top_level_reward, top_level_next_obs, top_level_done)


    def sample(self, batch_size, agents_status, top_level_next_action):
        """Sample transitions from the buffer"""
        # get the total num of transitions, these buffers should have same number of transitions
        total_num = len(self.buffers['agent_0'])
        indices = np.random.choice(total_num, size=batch_size, replace=True)
        
        # but only the reward and done of the current agent is needed in the calculation
        obs, act, reward, next_obs, cur_hour, done, next_act = {}, {}, {}, {}, {}, {}, {}
        for agent_id, buffer in self.buffers.items():
            # sample experience from the buffer
            o, a, r, cur_hr, n_o, d = buffer.sample(indices)
            obs[agent_id] = o
            act[agent_id] = a
            reward[agent_id] = r
            cur_hour[agent_id] = cur_hr
            next_obs[agent_id] = n_o
            done[agent_id] = d
            # calculate next_action using target_network and next_state
            next_act[agent_id] = self.agents[agent_id].target_action(n_o, agents_status[agent_id], top_level_next_action)
        
        return obs, act, reward, cur_hour, next_obs, done, next_act
    
    def top_level_sample(self, batch_size):
        """Sample transitions from the top level buffer"""
        top_level_obs, top_level_act, top_level_reward, top_level_next_obs, top_level_done = self.top_level_buffer.sample(batch_size)
        top_level_next_act = self.top_level_agent.target_action(top_level_next_obs)
        return top_level_obs, top_level_act, top_level_reward, top_level_next_obs, top_level_done, top_level_next_act


    def select_action(self, obs, top_level_action, agents_status):
        """Select action using GB_MARL"""

        actions = {}
        for agent, o in obs.items():
            o = torch.from_numpy(o).unsqueeze(0).float()
            a = self.agents[agent].action(o, agents_status[agent], top_level_action)  # torch.Size([1, action_size])
            # NOTE that the output is a tensor, convert it to int before input to the environment
            # actions[agent] = a.squeeze(0).argmax().item()
            actions[agent] = a.squeeze(0).item()
            # self.logger.info(f'{agent} action: {actions[agent]}')
        
        return actions
    
    
    def select_top_level_action(self, global_observation):
        """Select top level action using GB_MARL"""
        top_level_obs = global_observation
        top_level_action = self.top_level_agent.action(torch.from_numpy(top_level_obs).unsqueeze(0).float()) # select top level action
        top_level_action = int(top_level_action.item()) # get the top level action
        
        return top_level_action
    
    def pass_high_obs_to_low_obs(self, obs, top_level_act, top_level_obs, agents):
        """Pass observation to the agents"""
        top_level_obs_tensor, top_level_act_tensor = torch.tensor(top_level_obs, dtype=torch.float32).to(self.device), torch.tensor(top_level_act, dtype=torch.float32).to(self.device)
        top_level_obs_tensor, top_level_act_tensor = top_level_obs_tensor.view(-1, len(top_level_obs)), top_level_act_tensor.view(-1, 1)
        top_level_critic_value = self.top_level_agent.critic_value(top_level_obs_tensor, top_level_act_tensor)
        top_level_critic_scalar = top_level_critic_value.mean().item()
        
        for agent_id in agents:
            obs[agent_id][6] = top_level_act
            obs[agent_id][7] = top_level_critic_scalar
        
        return obs

    
    def pass_low_obs_to_high_obs(self, next_global_observation, obs, actions, agents):
        """Pass low observation to the top level agent"""
        low_obs_list, low_act_list = [], []
        # get the observation and action of the low level agents
        for agent_id in agents:
            low_obs_tensor = torch.tensor(obs[agent_id], dtype=torch.float32).to(self.device).unsqueeze(0)
            low_act_tensor = torch.tensor(actions[agent_id], dtype=torch.float32).to(self.device).unsqueeze(0)
            low_obs_tensor = low_obs_tensor.view(1, -1)
            low_act_tensor = low_act_tensor.view(1, -1)
            low_obs_list.append(low_obs_tensor)
            low_act_list.append(low_act_tensor)
        
        # calculate the critic value of the low level agents
        low_critic_values = self.agents[agent_id].critic_value(low_obs_list, low_act_list)
        critic_values = [value.item() for value in low_critic_values]
        
        # calculate the average and standard deviation of the critic values
        avg_critic_value = np.mean(critic_values)
        std_critic_value = np.std(critic_values)
        next_global_observation[8] = avg_critic_value
        next_global_observation[9] = std_critic_value
        
        return next_global_observation
    
    def learn(self, batch_size, top_level_batch_size, gamma, agents_status, step):
        """Learn from the replay buffer"""
        
        # top level agent
        top_level_obs, top_level_act, top_level_reward, top_level_next_obs, top_level_done, top_level_next_act = \
            self.top_level_sample(top_level_batch_size) # sample transitions from the top level buffer
        
        # update the top level critic 
        top_level_critic_value = self.top_level_agent.critic_value(top_level_obs, top_level_act) # calculate the value of the critic network

        # calculate top level target critic value
        next_top_target_critic_value = self.top_level_agent.target_critic_value(top_level_next_obs, top_level_next_act) # calculate the value of the target critic network
        
        # calculate top level target value
        top_target_value = top_level_reward + gamma * next_top_target_critic_value * (1 - top_level_done) # calculate the target value
        
        # calculate top level critic loss
        top_critic_loss = F.mse_loss(top_level_critic_value, top_target_value.detach(), reduction='mean') # calculate the critic loss
        self.top_level_agent.update_critic(top_critic_loss) # update the critic network
        
        # update the top level actor
        top_action, top_logits = self.top_level_agent.action(top_level_obs, model_out=True) # select action
        top_level_act = top_action
        top_level_actor_loss = -self.top_level_agent.critic_value(top_level_obs, top_level_act).mean() # calculate the actor loss using advantage
        top_level_actor_loss_pse = torch.pow(top_logits, 2).mean() # calculate the actor loss pse
        self.top_level_agent.update_actor(top_level_actor_loss + 1e-3 * top_level_actor_loss_pse) # update the actor network
        # self.logger.info(f'Top Level Agent: critic loss: {top_critic_loss.item()}, actor loss: {top_level_actor_loss.item()}') 
        
        wandb.log({
            "top agent critic loss:": top_critic_loss.item(),
            "top agent actor loss": top_level_actor_loss.item()
        }, step=step)

        
        top_level_next_action = int(torch.mean(top_level_next_act).item() >= 0.5)
        top_level_action = int(torch.mean(top_level_act).item() >= 0.5)
        for agent_id, agent in self.agents.items():
            obs, act, reward, current_hour, next_obs, done, next_act = self.sample(batch_size, agents_status, top_level_next_action)
            
            # update critic
            critic_value = agent.critic_value(list(obs.values()), list(act.values()))

            # calculate target critic value
            next_target_critic_value = agent.target_critic_value(list(next_obs.values()), list(next_act.values()))
            
            # calculate target value
            target_value = reward[agent_id] + gamma * next_target_critic_value * (1 - done[agent_id])

            # calculate critic loss
            critic_loss = F.mse_loss(critic_value, target_value.detach(), reduction='mean')
            agent.update_critic(critic_loss) # update critic

            # update actor
            # action of the current agent is calculated using its actor
            action, logits = agent.action(obs[agent_id], agents_status[agent_id], top_level_action, model_out=True)
            act[agent_id] = action
            agent_critic_value = agent.critic_value(list(obs.values()), list(act.values()))
            lcb = self.LCB(agent_critic_value, act[agent_id], obs[agent_id], current_hour[agent_id])
            actor_loss = -lcb.mean()
            actor_loss_pse = torch.pow(logits, 2).mean()
            agent.update_actor(actor_loss + 1e-3 * actor_loss_pse) # update actor
            # self.logger.info(f'{agent_id}: critic loss: {critic_loss.item()}, actor loss: {actor_loss.item()}')
            
            wandb.log({
                "low agent critic loss:": critic_loss.item(),
                "low agent actor loss": actor_loss.item()
            }, step=step)
        
    def LCB(self, actions_critics_values, actions, obs_agent_id, current_hour_agent_id, epsilon=1e-8):
        """Calculate the LCB"""

        def clamp_and_check_nan(tensor, min_val, device):
            """Clamps the tensor to avoid invalid values and checks for NaN."""
            tensor = tensor.clamp(min=min_val)
            return torch.where(torch.isnan(tensor), torch.tensor(0.0, device=device), tensor)
        
        actions = actions.squeeze(1)
        SoC_d, SoC_t = obs_agent_id[:, 1], obs_agent_id[:, 0]
        t_d, t_a = obs_agent_id[:, 2], obs_agent_id[:, 3]

        # Add epsilon to avoid log(0) and division by zero issues
        safe_actions = actions + epsilon

        # Clamp and check for NaN
        SoC_diff = clamp_and_check_nan(SoC_d - SoC_t, epsilon, actions.device)
        time_diff = clamp_and_check_nan(t_d - t_a, epsilon, actions.device)
        time_remaining = (t_d - current_hour_agent_id).clamp(min=0)

        # Calculate the LCB term
        rho = 1
        sqrt_term = clamp_and_check_nan(torch.sqrt(SoC_diff * (time_remaining / time_diff)), epsilon, actions.device)
        lcb = actions_critics_values - rho * torch.abs(torch.log2(safe_actions)) * sqrt_term

        return lcb

        
    def update_target(self, tau):
        """Update the target network"""
        def soft_update(from_network, to_network):
            """Soft update the target network"""
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                # Update the target network parameters using the soft update rule
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        # update the target network for all agents
        for agent in self.agents.values():
            soft_update(agent.actor, agent.target_actor)
            soft_update(agent.critic, agent.target_critic)
            
        # update the target network for the top level agent
        soft_update(self.top_level_agent.actor, self.top_level_agent.target_actor)  
        soft_update(self.top_level_agent.critic, self.top_level_agent.target_critic)

    def save(self, reward):
        model_state = {
            'agent_actor': {name: agent.actor.state_dict() for name, agent in self.agents.items()},
            'agent_critic': {name: agent.critic.state_dict() for name, agent in self.agents.items()},
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
        
        for agent_id, agent in instance.agents.items():
            agent.actor.load_state_dict(data['agent_actor'][agent_id])
            agent.critic.load_state_dict(data['agent_critic'][agent_id])
        
        instance.top_level_agent.actor.load_state_dict(data['top_level_agent_actor'])
        instance.top_level_agent.critic.load_state_dict(data['top_level_agent_critic'])
        
        return instance  # return the instance