import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from MADDPG import MADDPG
from logger_config import configured_logger as logger
from maddpg_parameter import parse_args, get_env
from utils import prepare_ev_request_data, create_result_dir, prepare_ev_departure_data, plot_training_results, plot_global_training_results
from tqdm import tqdm
from dotenv import load_dotenv
import os
load_dotenv()

alpha = ALPHA = float(os.getenv('REWARD_ALPHA'))
beta = BETA = float(os.getenv('REWARD_BETA'))

# Define the start and end date of the EV request data
start_date = START_DATE = os.getenv('START_DATETIME', '2018-07-01')
end_date = END_DATE = os.getenv('END_DATETIME', '2018-10-01')

# Define the start and end date of the EV request data without year
start_date_without_year = START_DATE[5:]  # Assuming the format is 'YYYY-MM-DD'
end_date_without_year = END_DATE[5:]  # Assuming the format is 'YYYY-MM-DD'

# Define the start and end datetime of the EV request data
start_time = START_TIME = datetime.strptime(start_date, '%Y-%m-%d')
end_time = END_TIME = datetime.strptime(end_date, '%Y-%m-%d')

# Define the number of agents
num_agents = NUM_AGENTS = int(os.getenv('NUM_AGENTS'))

# Define the path to the EV request data
parking_data_path = PARKING_DATA_PATH = f'../Dataset/Sim_Parking/ev_parking_data_from_2018-07-01_to_2018-12-31_{NUM_AGENTS}.csv'

# Define the directory name to save the result
# dir_name = DIR_NAME = 'GB-MARL-Discrete'
dir_name = DIR_NAME = os.getenv('DIR_NAME', 'GB-MARL-TEST')

if __name__ == '__main__':
    
    # parse arguments
    args = parse_args()
    
    # Define the start and end date of the EV request data
    ev_request_dict = prepare_ev_request_data(parking_data_path, start_date, end_date)
    ev_departure_dict = prepare_ev_departure_data(parking_data_path, start_date, end_date)
    
    # create environment
    env, dim_info, top_dim_info = get_env(num_agents, start_time, end_time)

    # create a new folder to save the result
    result_dir = create_result_dir(f'{DIR_NAME}_alpha{alpha}_beta{beta}_num{NUM_AGENTS}') 
    # result_dir = create_result_dir(f'{DIR_NAME}') 
    
    # create MADDPG agent
    maddpg = MADDPG(dim_info, top_dim_info, args.buffer_capacity, args.batch_size, 
                    args.top_level_buffer_capacity, args.top_level_batch_size, args.actor_lr, args.critic_lr, args.epsilon, args.sigma, result_dir) 

    step = 0  # global step counter
    agent_num = env.num_agents # number of agents
    
    # reward of each episode of each agent
    episode_rewards = {
        agent_id: np.zeros(args.episode_num) 
        for agent_id in env.agents
    }
    
    episode_global_rewards = np.zeros(args.episode_num)  # global reward of each episode
    
    # training
    for episode in tqdm(range(args.episode_num)):
        
        # decrease epsilon
        maddpg.change_top_level_agent_parameter(args.epsilon * (1 - episode / args.episode_num), args.sigma_decay)
        maddpg.top_level_agent.update_target_network(0.1)
        # reset the timestamp to the start time of the environment
        env.timestamp = env.start_time 
        obs, global_observation = env.reset()
        
        # agent reward of the current episode
        agent_reward = {
            agent_id: 0 for agent_id in env.agents
        }  

        curr_global_reward = 0  # global reward of the current episode
        
        while env.timestamp <= env.end_time:  
            
            # skip the time
            if env.timestamp.hour < 7 or env.timestamp.hour > 23:
                env.timestamp += timedelta(hours=1)
                continue
            
            # add EVs to the environment, if there are EVs that have arrived at the current time
            current_requests = ev_request_dict.get(env.timestamp, []) # get the EVs that have arrived at the current time
            for ev in current_requests:
                env.add_ev(ev['requestID'], 
                            ev['arrival_time'], 
                            ev['departure_time'], 
                            ev['initial_soc'], 
                            ev['departure_soc'])
                
                env.current_parking_number += 1 # increase the number of EVs in the environment
                        
            # Remove EVs that departed at the current time
            current_departures = ev_departure_dict.get(env.timestamp, [])
            for ev in current_departures:
                for agent_id, data in env.ev_data.items():
                    if ev['requestID'] == data['requestID']:
                        env.remove_ev(agent_id)
                        env.current_parking_number -= 1 # decrease the number of EVs in the environment
            
            # select action
            step += 1
            if step < args.random_steps:
                action = {}
                top_level_action = env.get_top_level_action_space().sample() # sample top level action
                # print(f'top_level_action: {top_level_action}')
                for agent_id in env.agents:
                    if env.agents_status[agent_id]:
                        # if the agent is connected 
                        if top_level_action == 0:
                            action[agent_id] = env.charging_action_space(agent_id).sample()
                            # print(f'charging action: {action[agent_id]}')
                        else:
                            action[agent_id] = env.discharging_action_space(agent_id).sample()
                            # print(f'discharging action: {action[agent_id]}')
                    else:
                        # if the agent is not connected
                        action[agent_id] = -1e10 # set the action to -1
            else:
                action, top_level_action = maddpg.select_action(obs, global_observation, env.agents_status) # select action using MADDPG
                # print(f'top_level_action: {top_level_action}, action: {action}')
            # print(f'top_level_action: {top_level_action}, action: {action}')
            # import time
            # time.sleep(1)
            # step the environment
            next_obs, next_global_observation, reward, global_reward, done, info = env.step(action, top_level_action, env.timestamp)

            # add experience to replay buffer
            maddpg.add(obs, global_observation, action, top_level_action, reward, global_reward, next_obs, next_global_observation, done, env.agents_status)
            
            # update reward
            for agent_id, r in reward.items():  
                agent_reward[agent_id] += r 
            
            curr_global_reward += global_reward
            
            # learn from the replay buffer
            if step >= args.random_steps and step % args.learn_interval == 0:  # learn every few steps
                maddpg.learn(args.batch_size, args.top_level_batch_size, args.gamma, env.agents_status) # learn from the replay buffer
                maddpg.update_target(args.tau) # update target network
                
            # update observation
            obs = next_obs
            global_observation = next_global_observation
            # print(global_observation)

            # if the current time is greater than or equal to the end time, break
            if env.timestamp >= env.end_time: 
                break

        
        # episode finishes
        for agent_id, r in agent_reward.items():  # record reward
            episode_rewards[agent_id][episode] = r
            
        episode_global_rewards[episode] = curr_global_reward
       
        if (episode + 1) % 100 == 0:  # print info every 100 episodes
            message = f'episode {episode + 1}, '
            sum_reward = 0
            for agent_id, r in agent_reward.items():  # record reward
                message += f'{agent_id}: {r:>4f}; '
                sum_reward += r
            message += f'sum reward: {sum_reward}'
            logger.bind(console=True).info(message)
            logger.bind(console=True).info(f'global reward: {curr_global_reward}')

    maddpg.save(episode_rewards)  # save model
    plot_training_results(episode_rewards, args, result_dir)
    plot_global_training_results(episode_global_rewards, args, result_dir)

    # save soc history and charging records
    soc_history_file = f'{result_dir}/soc_history.csv'
    charging_records_file = f'{result_dir}/charging_records.csv'
    env.soc_history.to_csv(soc_history_file, index=False)
    env.charging_records.to_csv(charging_records_file, index=False)
    
    # save load history
    load_history_df = pd.DataFrame(env.load_history)
    load_history_file = f'{result_dir}/building_loading_history.csv'
    load_history_df.to_csv(load_history_file, index=False)
    
    
    