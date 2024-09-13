import wandb
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from GB_MARL import GB_MARL
from logger_config import configured_logger as logger
from GB_MARL_parameter import parse_args, get_env
from utils import (prepare_ev_request_data, create_result_dir, 
                   prepare_ev_departure_data, plot_training_results, 
                   plot_global_training_results, prepare_ev_actual_departure_data, set_seed)
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

# Test data
test_start_date = TEST_START_DATE = os.getenv('TEST_START_DATETIME', '2018-10-01')
test_end_date = TEST_END_DATE = os.getenv('TEST_END_DATETIME', '2018-10-08')
test_start_time = TEST_START_TIME = datetime.strptime(test_start_date, '%Y-%m-%d')
test_end_time = TEST_END_TIME = datetime.strptime(test_end_date, '%Y-%m-%d')

# Define the number of agents
num_agents = NUM_AGENTS = int(os.getenv('NUM_AGENTS'))

# Define the path to the EV request data
parking_version = PARKING_VERSION = os.getenv('PARKING_VERSION')
parking_data_path = PARKING_DATA_PATH = f'../Dataset/Sim_Parking/ev_parking_data_v{PARKING_VERSION}_from_2018-07-01_to_2018-12-31_{NUM_AGENTS}.csv'

# Define the directory name to save the result
dir_name = DIR_NAME = f'GB-MARL-v2'
# dir_name = DIR_NAME = 'TEST'

if __name__ == '__main__':
   
    # set seed
    set_seed(42)
    
    # parse arguments
    args = parse_args()
    default_config = vars(args)
    default_config['alpha'] = alpha
    default_config['beta'] = beta
    default_config['start_date'] = start_date
    default_config['end_date'] = end_date
    default_config['test_start_date'] = test_start_date
    default_config['test_end_date'] = test_end_date
    default_config['num_agents'] = num_agents

    wandb.init(project=DIR_NAME, config=default_config)
    wandb.run.name = DIR_NAME
    wandb.run.save()
    
    # Define the start and end date of the EV request data
    ev_request_dict = prepare_ev_request_data(parking_data_path, start_date, end_date)
    ev_departure_dict = prepare_ev_departure_data(parking_data_path, start_date, end_date) \
        if parking_version == '0' else prepare_ev_actual_departure_data(parking_data_path, start_date, end_date)
    
    # create environment
    env, dim_info, top_dim_info = get_env(num_agents, start_time, end_time)

    # create a new folder to save the result
    result_dir = create_result_dir(f'{DIR_NAME}_alpha{alpha}_beta{beta}_num{NUM_AGENTS}_s{PARKING_VERSION}') 
    
    # create MADDPG agent
    gb_marl = GB_MARL(dim_info, top_dim_info, args.buffer_capacity, args.batch_size, 
                    args.top_level_buffer_capacity, args.top_level_batch_size, 
                    args.actor_lr, args.critic_lr, result_dir) 

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
                        ev_departure_time = ev['departure_time'] if parking_version == '0' \
                            else ev['actual_departure_time']
                        env.remove_ev(agent_id, ev_departure_time, env.timestamp)
                        env.current_parking_number -= 1 # decrease the number of EVs in the environment
            
            # select actions
            step += 1
            if step < args.random_steps:
                actions = {}
                top_level_action = env.get_top_level_action_space().sample() # sample top level actions
                obs = gb_marl.pass_high_obs_to_low_obs(obs, top_level_action, global_observation, env.agents)
                # print(f'top_level_action: {top_level_action}')
                for agent_id in env.agents:
                    if env.agents_status[agent_id]:
                        # if the agent is connected 
                        if top_level_action == 0:
                            actions[agent_id] = env.charging_action_space(agent_id).sample()
                            # print(f'charging actions: {actions[agent_id]}')
                        else:
                            actions[agent_id] = env.discharging_action_space(agent_id).sample()
                            # print(f'discharging actions: {actions[agent_id]}')
                    else:
                        # if the agent is not connected
                        actions[agent_id] = -1e10 # set the actions to -1
            else:
                top_level_action = gb_marl.select_top_level_action(global_observation)
                obs = gb_marl.pass_high_obs_to_low_obs(obs, top_level_action, global_observation, env.agents)
                actions = gb_marl.select_action(obs, top_level_action, env.agents_status) 
            
            # NOTE next_global_observation 是用 Q network 算出這個時間點的十個 actions 的平均值和十個 actions 的標準差
            # NOTE next_obs 直接用這個時間點的 global actions 和 critic value
            next_obs, next_global_observation, reward, global_reward, done, info = env.step(actions, env.timestamp)
            next_global_observation = gb_marl.pass_low_obs_to_high_obs(next_global_observation, obs, actions, env.agents)

            # add experience to replay buffer
            gb_marl.add(obs, global_observation, actions, top_level_action, reward, 
                       global_reward, next_obs, next_global_observation, env.timestamp.hour, done)
            
            # update reward
            for agent_id, r in reward.items():  
                agent_reward[agent_id] += r 
            
            # update global reward
            curr_global_reward += global_reward
            
            # learn from the replay buffer
            if step >= args.random_steps and step % args.learn_interval == 0:  # learn every few steps
                gb_marl.learn(args.batch_size, args.top_level_batch_size, args.gamma, env.agents_status, step) # learn from the replay buffer
                gb_marl.update_target(args.tau) # update target network
                
            # update observation
            # NOTE 用更新過的 Q network 算出這個時間點的十個 actions 的平均值和十個 actions 的標準差取代原本的。
            global_observation = next_global_observation 
            obs = next_obs

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
            wandb.log(
                {
                    "episode": episode + 1,
                    "sum_reward": sum_reward,
                    "global_reward": curr_global_reward
                }, step=step
            )

    gb_marl.save(episode_rewards)  # save model
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
    
    # ==========================
    # Execute testing phase
    # ==========================
    
    testing_ev_request_dict = prepare_ev_request_data(parking_data_path, test_start_date, test_end_date)
    testing_ev_departure_dict = prepare_ev_departure_data(parking_data_path, test_start_date, test_end_date) \
        if parking_version == '0' else prepare_ev_actual_departure_data(parking_data_path, test_start_date, test_end_date)
    
    # Create environment for testing (September 1st to September 7th)
    test_env, _, _ = get_env(num_agents, test_start_time, test_end_time)

    # Load the trained model
    loaded_model = GB_MARL.load(
                        dim_info=dim_info,
                        top_dim_info=top_dim_info,
                        actor_lr=args.actor_lr,
                        critic_lr=args.critic_lr,
                        res_dir=result_dir,
                        file=os.path.join(result_dir, 'model.pt')
                    )

    # Initialize testing variables
    test_results = []
    obs, global_observation = test_env.reset()
    
    while test_env.timestamp <= test_env.end_time:
        if test_env.timestamp.hour < 7 or test_env.timestamp.hour > 23:
            test_env.timestamp += timedelta(hours=1)
            continue
        
        current_requests = testing_ev_request_dict.get(test_env.timestamp, [])
        for ev in current_requests:
            test_env.add_ev(ev['requestID'], 
                            ev['arrival_time'], 
                            ev['departure_time'], 
                            ev['initial_soc'], 
                            ev['departure_soc'])
            test_env.current_parking_number += 1
        
        current_departures = testing_ev_departure_dict.get(test_env.timestamp, [])
        for ev in current_departures:
            for agent_id, data in test_env.ev_data.items():
                if ev['requestID'] == data['requestID']:
                    ev_departure_time = ev['departure_time'] if parking_version == '0' \
                            else ev['actual_departure_time']
                    test_env.remove_ev(agent_id, ev_departure_time, test_env.timestamp)
                    test_env.current_parking_number -= 1
        
        top_level_action = gb_marl.select_top_level_action(global_observation)
        actions = loaded_model.select_action(obs, top_level_action, test_env.agents_status)
        next_obs, next_global_observation, reward, global_reward, done, info = test_env.step(actions, test_env.timestamp)

        obs = next_obs
        global_observation = next_global_observation
    
    # Save testing soc history and charging records
    test_soc_history_file = f'{result_dir}/test_soc_history.csv'
    test_charging_records_file = f'{result_dir}/test_charging_records.csv'
    test_env.soc_history.to_csv(test_soc_history_file, index=False)
    test_env.charging_records.to_csv(test_charging_records_file, index=False)
    
    # Save testing load history
    test_load_history_df = pd.DataFrame(test_env.load_history)
    test_load_history_file = f'{result_dir}/test_building_loading_history.csv'
    test_load_history_df.to_csv(test_load_history_file, index=False)
    
    print(f'Test results and histories saved to {result_dir}')
    
    
    