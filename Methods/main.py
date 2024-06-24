import argparse
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from EVBuildingEnvMADDPG import EVBuildingEnv
from datetime import datetime
from MADDPG import MADDPG
from logger_config import configured_logger as logger


# Define the start and end date of the EV request data
start_date = START_DATE = '2018-07-01'
end_date = END_DATE = '2018-12-31'

# Define the start and end time of the EV request data
start_time = START_TIME = datetime(2018, 7, 1)
end_time = END_TIME = datetime(2018, 12, 31)

# Define the number of agents
num_agents = NUM_AGENTS = 10
parking_data_path = PARKING_DATA_PATH = '../Dataset/Sim_Parking/ev_parking_data_from_2018-07-01_to_2018-12-31.csv'


def get_env(num_agents, start_time, end_time):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = EVBuildingEnv(num_agents, start_time, end_time)
    
    new_env.reset()
    _dim_info = {}
    for agent_id in new_env.agents:
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
        _dim_info[agent_id].append(new_env.action_space(agent_id).size())

    return new_env, _dim_info


if __name__ == '__main__':
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode_num', type=int, default=30000, help='total episode num during training procedure') # episode number
    parser.add_argument('--episode_length', type=int, default=25, help='steps per episode') # episode length
    parser.add_argument('--learn_interval', type=int, default=100, help='steps interval between learning time') # learn interval
    parser.add_argument('--random_steps', type=int, default=5e4, help='random steps before the agent start to learn') # random steps
    parser.add_argument('--tau', type=float, default=0.02, help='soft update parameter') # tau
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor') # gamma
    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='capacity of replay buffer') # buffer capacity
    parser.add_argument('--batch_size', type=int, default=1024, help='batch-size of replay buffer') # batch size
    parser.add_argument('--actor_lr', type=float, default=0.01, help='learning rate of actor') # learning rate of actor
    parser.add_argument('--critic_lr', type=float, default=0.01, help='learning rate of critic') # learning rate of critic
    args = parser.parse_args()
    
    # Define the start and end date of the EV request data
    ev_request_data = pd.read_csv(parking_data_path, parse_dates=['arrival_time', 'departure_time']) 
    ev_request_data = ev_request_data[(ev_request_data['date'] >= start_date) & (ev_request_data['date'] < end_date)].copy()
    ev_request_data['date'] = pd.to_datetime(ev_request_data['date']).dt.date # convert date to datetime object
    ev_request_dict = ev_request_data.groupby(ev_request_data['arrival_time']).apply(lambda x: x.to_dict(orient='records')).to_dict()
    
    # create folder to save result
    env_dir = os.path.join('../Result', 'EVBuildingEnv')
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    total_files = len([file for file in os.listdir(env_dir)]) # current number of files
    result_dir = os.path.join(env_dir, f'{total_files + 1}') # create a new folder, which is the number of current files + 1
    os.makedirs(result_dir) # create the new folder

    # create environment
    env, dim_info = get_env(num_agents, start_time, end_time) 
    maddpg = MADDPG(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr, result_dir) # create MADDPG agent

    step = 0  # global step counter
    agent_num = env.num_agents # number of agents
    
    # reward of each episode of each agent
    episode_rewards = {agent_id: np.zeros(args.episode_num) for agent_id in env.agents}
    for episode in range(args.episode_num):
        env.timestamp = env.start_time
        obs = env.reset()
        agent_reward = {agent_id: 0 for agent_id in env.agents}  # agent reward of the current episode
        while env.timestamp <= env.end_time: 
            if env.timestamp.hour == 0:
                logger.bind(console=True).info(f'current time: {env.timestamp}, episode: {episode + 1}, Reward: {sum(agent_reward.values())}')

            env.count = 0 # count the number of EVs that have arrived
            current_requests = ev_request_dict.get(env.timestamp, []) # get the EVs that have arrived at the current time

            # add EVs to the environment, if there are EVs that have arrived at the current time
            if current_requests:
                for ev in current_requests:
                    env.add_ev(ev['requestID'], 
                               ev['arrival_time'], 
                               ev['departure_time'], 
                               ev['initial_soc'], 
                               ev['departure_soc'])
                    
                    env.current_parking_number += 1 # increase the number of EVs in the environment
                    env.count += 1 # increase the number of EVs that have arrived
            
            step += 1
            if step < args.random_steps: 
                action = {}
                for agent_id in env.agents:
                    if env.agents_status[agent_id] and env.timestamp > env.ev_data[agent_id]['arrival_time']:
                        # if the agent is connected and the current time is greater than the arrival time of the EV
                        action[agent_id] = env.action_space(agent_id).sample()
                    elif env.agents_status[agent_id] and env.timestamp == env.ev_data[agent_id]['arrival_time']:
                        # if the agent is connected and the current time is equal to the arrival time of the EV
                        action[agent_id] = -1  
                    else:
                        # if the agent is not connected
                        action[agent_id] = 0
            else:
                # select action
                action = maddpg.select_action(obs)
            
            # step the environment
            next_obs, reward, done, info = env.step(action, env.timestamp)

            # add experience to replay buffer
            maddpg.add(obs, action, reward, next_obs, done)
            
            # update reward
            for agent_id, r in reward.items():  
                agent_reward[agent_id] += r
            
            # learn from the replay buffer
            if step >= args.random_steps and step % args.learn_interval == 0:  # learn every few steps
                maddpg.learn(args.batch_size, args.gamma) # learn from the replay buffer
                maddpg.update_target(args.tau) # update target network
                
            # update observation
            obs = next_obs

            # if the current time is greater than or equal to the end time, break
            if env.timestamp >= env.end_time: 
                break

        
        # episode finishes
        for agent_id, r in agent_reward.items():  # record reward
            episode_rewards[agent_id][episode] = r
       
        if (episode + 1) % 100 == 0:  # print info every 100 episodes
            message = f'episode {episode + 1}, '
            sum_reward = 0
            for agent_id, r in agent_reward.items():  # record reward
                message += f'{agent_id}: {r:>4f}; '
                sum_reward += r
            message += f'sum reward: {sum_reward}'
            logger.bind(console=True).info(message)

    maddpg.save(episode_rewards)  # save model


    def get_running_reward(arr: np.ndarray, window=100):
        """calculate the running reward, i.e. average of last `window` elements from rewards"""
        running_reward = np.zeros_like(arr)
        effective_window = min(window, len(arr))  # ensure the window is not greater than the length of rewards
        for i in range(effective_window - 1):
            running_reward[i] = np.mean(arr[:i + 1])
        for i in range(effective_window - 1, len(arr)):
            running_reward[i] = np.mean(arr[i - effective_window + 1:i + 1])
        return running_reward


    # training finishes, plot reward
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    for agent_id, reward in episode_rewards.items():
        ax.plot(x, reward, label=agent_id)
        ax.plot(x, get_running_reward(reward))
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    title = f'training result of maddpg solve EVBuildingEnv'
    ax.set_title(title)
    plt.savefig(os.path.join(result_dir, title))

    # save soc history and charging records
    soc_history_file = f'{result_dir}/soc_history.csv'
    charging_records_file = f'{result_dir}/charging_records.csv'
    env.soc_history.to_csv(soc_history_file, index=False)
    env.charging_records.to_csv(charging_records_file, index=False)
    
    # save load history
    load_history_df = pd.DataFrame(env.load_history)
    load_history_file = f'{result_dir}/load_history.csv'
    load_history_df.to_csv(load_history_file, index=False)