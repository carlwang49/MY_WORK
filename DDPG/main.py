import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from logger_config import configured_logger as logger
from EVBuildingEnv import EVBuildingEnv
from datetime import datetime, timedelta
from utils import prepare_ev_request_data, prepare_ev_departure_data, create_result_dir, get_running_reward
from tqdm import tqdm
import pandas as pd

# TODO: 合併 start_time, start_date, end_time, end_date

# Define the start and end date of the EV request data
start_date = START_DATE = '2018-07-01'
end_date = END_DATE = '2018-09-30'

# Define the start and end time of the EV request data
start_time = START_TIME = datetime(2018, 7, 1)
end_time = END_TIME = datetime(2018, 9, 30)

# Define the number of agents
num_agents = NUM_AGENTS = 30 
parking_data_path = PARKING_DATA_PATH = f'../Dataset/Sim_Parking/ev_parking_data_from_2018-07-01_to_2018-12-31_{num_agents}.csv'

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, action_dim)

    def forward(self, s):
        s = F.relu(self.l1(s))
        s = F.relu(self.l2(s))
        a = self.max_action * torch.tanh(self.l3(s))  # [-max,max]
        return a


class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, 1)

    def forward(self, s, a):
        q = F.relu(self.l1(torch.cat([s, a], 1)))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim):
        self.max_size = int(1e6)
        self.count = 0
        self.size = 0
        self.s = np.zeros((self.max_size, state_dim))
        self.a = np.zeros((self.max_size, action_dim))
        self.r = np.zeros((self.max_size, 1))
        self.s_ = np.zeros((self.max_size, state_dim))
        self.dw = np.zeros((self.max_size, 1))

    def store(self, s, a, r, s_, dw):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.count = (self.count + 1) % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.size = min(self.size + 1, self.max_size)  # Record the number of  transitions

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)  # Randomly sampling
        batch_s = torch.tensor(self.s[index], dtype=torch.float)
        batch_a = torch.tensor(self.a[index], dtype=torch.float)
        batch_r = torch.tensor(self.r[index], dtype=torch.float)
        batch_s_ = torch.tensor(self.s_[index], dtype=torch.float)
        batch_dw = torch.tensor(self.dw[index], dtype=torch.float)

        return batch_s, batch_a, batch_r, batch_s_, batch_dw


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.hidden_width = 256  # The number of neurons in hidden layers of the neural network
        self.batch_size = 256  # batch size
        self.GAMMA = 0.99  # discount factor
        self.TAU = 0.005  # Softly update the target network
        self.lr = 3e-4  # learning rate

        self.actor = Actor(state_dim, action_dim, self.hidden_width, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, action_dim, self.hidden_width)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.MseLoss = nn.MSELoss()

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a = self.actor(s).data.numpy().flatten()
        return a

    def learn(self, relay_buffer):
        batch_s, batch_a, batch_r, batch_s_, batch_dw = relay_buffer.sample(self.batch_size)  # Sample a batch

        # Compute the target Q
        with torch.no_grad():  # target_Q has no gradient
            Q_ = self.critic_target(batch_s_, self.actor_target(batch_s_))
            target_Q = batch_r + self.GAMMA * (1 - batch_dw) * Q_

        # Compute the current Q and the critic loss
        current_Q = self.critic(batch_s, batch_a)
        critic_loss = self.MseLoss(target_Q, current_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute the actor loss
        actor_loss = -self.critic(batch_s, self.actor(batch_s)).mean()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)


def evaluate_policy(env_evaluate: EVBuildingEnv, agent, test_episodes, ev_request_dict, ev_departure_dict):
    
    evaluate_reward = 0
    for _ in range(test_episodes):
        
        s = env_evaluate.reset()
        episode_reward = 0
        
        while env_evaluate.timestamp <= env_evaluate.end_time: 
            
            if env_evaluate.timestamp.hour < 7 or env_evaluate.timestamp.hour > 23:
                env_evaluate.timestamp += timedelta(hours=1)
                continue
            # add EVs to the environment, if there are EVs that have arrived at the current time
            current_requests = ev_request_dict.get(env_evaluate.timestamp, []) # get the EVs that have arrived at the current time
            if current_requests:
                for ev in current_requests:
                    env_evaluate.add_ev(ev['requestID'], 
                            ev['arrival_time'], 
                            ev['departure_time'], 
                            ev['initial_soc'], 
                            ev['departure_soc'])
                    
                    env_evaluate.current_parking_number += 1 # increase the number of EVs in the environment
                        
            # Remove EVs that departed at the current time
            current_departures = ev_departure_dict.get(env_evaluate.timestamp, [])
            if current_departures:
                for ev in current_departures:
                    request_id = ev['requestID']
                    for agent_id, data in env_evaluate.ev_data.items():
                        if data['requestID'] == request_id:
                            env_evaluate.remove_ev(agent_id)
                            env_evaluate.current_parking_number -= 1
                            break 
            
            a = agent.choose_action(s)  # We do not add noise when evaluating
            s_, r, done, _ = env_evaluate.step(a, env_evaluate.timestamp)
            episode_reward += r
            s = s_
            
        evaluate_reward += episode_reward

    return int(evaluate_reward / test_episodes)




if __name__ == '__main__':
    
    # Define the start and end date of the EV request data
    ev_request_dict = prepare_ev_request_data(parking_data_path, start_date, end_date)
    ev_departure_dict = prepare_ev_departure_data(parking_data_path, start_date, end_date)
    
    # create environment
    env = EVBuildingEnv(num_agents, start_time, end_time)
    env_evaluate = EVBuildingEnv(num_agents, start_time, end_time)  # When evaluating the policy, we need to rebuild an environment
    
    # Set random seed
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high)
    min_action = float(env.action_space.low)
    
    # create a new folder to save the result
    result_dir = create_result_dir('DDPG') 

    logger.bind(console=True).info("state_dim={}".format(state_dim))
    logger.bind(console=True).info("action_dim={}".format(action_dim))
    logger.bind(console=True).info("max_action={}".format(max_action))

    agent = DDPG(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    noise_std = 0.1 * max_action  # the std of Gaussian noise for exploration
    max_train_steps = 3e6  # Maximum number of training steps
    random_steps = 25e3  # Take the random actions in the beginning for the better exploration
    update_freq = 50  # Take 50 steps,then update the networks 50 times
    evaluate_freq = 1e3  # Evaluate the policy every 'evaluate_freq' steps
    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training
    episode_num = 3000
    test_episodes = 5  # The number of episodes when evaluating the policy
    
    for episode in tqdm(range(episode_num)):
        episode_steps = 0
        s = env.reset()
        while env.timestamp <= env.end_time:  
            
            if env.timestamp.hour < 7 or env.timestamp.hour > 23:
                env.timestamp += timedelta(hours=1)
                continue

            # add EVs to the environment, if there are EVs that have arrived at the current time
            current_requests = ev_request_dict.get(env.timestamp, []) # get the EVs that have arrived at the current time
            if current_requests:
                for ev in current_requests:
                    env.add_ev(ev['requestID'], 
                               ev['arrival_time'], 
                               ev['departure_time'], 
                               ev['initial_soc'], 
                               ev['departure_soc'])
                    
                    env.current_parking_number += 1 # increase the number of EVs in the environment
                        
            # Remove EVs that departed at the current time
            current_departures = ev_departure_dict.get(env.timestamp, [])
            if current_departures:
                for ev in current_departures:
                    for agent_id, data in env.ev_data.items():
                        if ev['requestID'] == data['requestID']:
                            env.remove_ev(agent_id)
                            env.current_parking_number -= 1
            
            episode_steps += 1
            if total_steps < random_steps:  # Take the random actions in the beginning for the better exploration
                a = env.action_space.sample(env.agents, env.agents_status)
            else:
                # Add Gaussian noise to actions for exploration
                a = agent.choose_action(s)
                a = (a + np.random.normal(0, noise_std, size=action_dim)).clip(min_action, max_action)
            s_, r, done, _ = env.step(a, env.timestamp)
                
            replay_buffer.store(s, a, r, s_, dw=False)  # Store the transition
            s = s_

            # Take 50 steps,then update the networks 50 times
            if total_steps >= random_steps and total_steps % update_freq == 0:
                for _ in range(update_freq):
                    agent.learn(replay_buffer)

            # Evaluate the policy every 'evaluate_freq' steps
            if (total_steps + 1) % evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(env_evaluate, agent, test_episodes, ev_request_dict, ev_departure_dict)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{}".format(evaluate_num, evaluate_reward))
                # Save the rewards
                if evaluate_num % 10 == 0:
                    np.save(result_dir, np.array(evaluate_rewards))

            total_steps += 1
        
    # save soc history and charging records
    soc_history_file = f'{result_dir}/soc_history.csv'
    charging_records_file = f'{result_dir}/charging_records.csv'
    env.soc_history.to_csv(soc_history_file, index=False)
    env.charging_records.to_csv(charging_records_file, index=False)
    
    # save load history
    load_history_df = pd.DataFrame(env.load_history)
    load_history_file = f'{result_dir}/building_loading_history.csv'
    load_history_df.to_csv(load_history_file, index=False)