import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from MADDPG import MADDPG
from logger_config import configured_logger as logger
from maddpg_parameter import parse_args, get_env
from utils import prepare_ev_request_data, create_result_dir, get_running_reward, prepare_ev_departure_data
from tqdm import tqdm

# TODO: 合併 start_time, start_date, end_time, end_date

# Define the start and end date of the EV request data
start_date = START_DATE = '2018-07-01'
end_date = END_DATE = '2018-09-02'

# Define the start and end time of the EV request data
start_time = START_TIME = datetime(2018, 7, 1)
end_time = END_TIME = datetime(2018, 7, 2)

# Define the number of agents
num_agents = NUM_AGENTS = 10 
parking_data_path = PARKING_DATA_PATH = f'../Dataset/Sim_Parking/ev_parking_data_from_2018-07-01_to_2018-12-31_{num_agents}.csv'


if __name__ == '__main__':
    
    # parse arguments
    args = parse_args()
    
    # Define the start and end date of the EV request data
    ev_request_dict = prepare_ev_request_data(parking_data_path, start_date, end_date)
    ev_departure_dict = prepare_ev_departure_data(parking_data_path, start_date, end_date)
    
    # create a new folder to save the result
    result_dir = create_result_dir('TEST') 
    
    # create environment
    env, dim_info = get_env(num_agents, start_time, end_time) 
    maddpg = MADDPG(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr, result_dir) # create MADDPG agent

    step = 0  # global step counter
    agent_num = env.num_agents # number of agents
    
    # reward of each episode of each agent
    episode_rewards = {
        agent_id: np.zeros(args.episode_num) 
        for agent_id in env.agents
    }
    
    # training
    for episode in tqdm(range(args.episode_num)):
        
        # reset the timestamp to the start time of the environment
        env.timestamp = env.start_time 
        obs = env.reset()
        
        # agent reward of the current episode
        agent_reward = {
            agent_id: 0 for agent_id in env.agents
        }  
        
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
            
            # select action
            step += 1
            if step < args.random_steps:
                action = {}
                for agent_id in env.agents:
                    if env.agents_status[agent_id]:
                        # if the agent is connected 
                        action[agent_id] = env.action_space(agent_id).sample()
                    else:
                        # if the agent is not connected
                        action[agent_id] = -1 # set the action to -1
            else:
                action = maddpg.select_action(obs, env.agents_status) # select action using MADDPG

            # step the environment
            next_obs, reward, done, info = env.step(action, env.timestamp)

            # add experience to replay buffer
            maddpg.add(obs, action, reward, next_obs, done)
            
            # update reward
            for agent_id, r in reward.items():  
                agent_reward[agent_id] += r
            
            # learn from the replay buffer
            if step >= args.random_steps and step % args.learn_interval == 0:  # learn every few steps
                maddpg.learn(args.batch_size, args.gamma, env.agents_status) # learn from the replay buffer
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
    load_history_file = f'{result_dir}/building_loading_history.csv'
    load_history_df.to_csv(load_history_file, index=False)