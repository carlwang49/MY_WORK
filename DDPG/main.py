import numpy as np
from logger_config import configured_logger as logger
from EVBuildingEnv import EVBuildingEnv
from datetime import datetime, timedelta
from utils import (prepare_ev_request_data, prepare_ev_departure_data, prepare_ev_actual_departure_data,
                   create_result_dir, plot_training_results, set_seed)
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from DDPG import DDPG
from NetWork import ReplayBuffer
import pickle
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
dir_name = DIR_NAME = 'DDPG'

# Define hyperparameters
random_steps = RANDOM_STEPS  = int(float(os.getenv('RANDOM_STEPS')))  # Take the random actions in the beginning for the better exploration
episode_num = EPISODE_NUM = int(os.getenv('NUMBER_OF_EPISODES'))
update_freq = UPDATE_FREQ = int(os.getenv('LEARN_INTERVAL'))
episode_rewards = EPISODE_REWARDS = defaultdict(int)  # Record the rewards during the evaluating

if __name__ == '__main__':
    
    # set seed
    set_seed(30)
    
    # Define the start and end date of the EV request data
    ev_request_dict = prepare_ev_request_data(parking_data_path, start_date, end_date)
    ev_departure_dict = prepare_ev_departure_data(parking_data_path, start_date, end_date) \
        if parking_version == '0' else prepare_ev_actual_departure_data(parking_data_path, start_date, end_date)
    
    # create environment
    env = EVBuildingEnv(num_agents, start_time, end_time)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high)
    min_action = float(env.action_space.low)
    
    # create a new folder to save the result
    result_dir = create_result_dir(f'{DIR_NAME}_alpha{ALPHA}_beta{BETA}_{NUM_AGENTS}_sim_v{PARKING_VERSION}') 

    agent = DDPG(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    noise_std = 0.1 * max_action  # the std of Gaussian noise for exploration
    episode_steps = 0
    
    for episode in tqdm(range(episode_num)):
        
        episode_reward = 0
        s = env.reset()
        while env.timestamp <= env.end_time:  
            
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
                        env.current_parking_number -= 1
                        
            episode_steps += 1
            if episode_steps < random_steps:  # Take the random actions in the beginning for the better exploration
                a = env.action_space.sample(env.agents, env.agents_status)
            else:
                # Add Gaussian noise to actions for exploration
                a = agent.choose_action(s)
                a = (a + np.random.normal(0, noise_std, size=action_dim)).clip(min_action, max_action)
            s_, r, done, _ = env.step(a, env.timestamp)

            # update the episode reward
            episode_reward += r
            
            replay_buffer.store(s, a, r, s_, dw=False)  # Store the transition
            s = s_

            # Take 50 steps,then update the networks 50 times
            if episode_steps >= random_steps and episode_steps % update_freq == 0:
                for _ in range(update_freq):
                    agent.learn(replay_buffer)
        
        # episode is over
        episode_rewards[episode + 1] += episode_reward
        if (episode + 1) % 100 == 0:
            logger.bind(console=True).info(f"Episode: {episode + 1}, Reward: {episode_reward}")
            
    # Save the rewards data
    with open(os.path.join(result_dir, 'rewards.pkl'), 'wb') as f:
        pickle.dump(episode_rewards, f)

    # Plot the training results
    plot_training_results(episode_rewards, episode_num, result_dir)
    
    # save soc history and charging records
    soc_history_file = f'{result_dir}/soc_history.csv'
    charging_records_file = f'{result_dir}/charging_records.csv'
    env.soc_history.to_csv(soc_history_file, index=False)
    env.charging_records.to_csv(charging_records_file, index=False)
    
    # save load history
    load_history_df = pd.DataFrame(env.load_history)
    load_history_file = f'{result_dir}/building_loading_history.csv'
    load_history_df.to_csv(load_history_file, index=False)
    
    # Save the model after training
    model_save_path = os.path.join(result_dir, 'ddpg_model.pth')
    agent.save(model_save_path)

    # ==========================
    # Execute testing phase
    # ==========================
    
    # Load the trained model for testing
    model_load_path = os.path.join(result_dir, 'ddpg_model.pth')
    agent.load(model_load_path)

    # Create a new environment for the testing phase
    test_env = EVBuildingEnv(num_agents, test_start_time, test_end_time)

    # Prepare the EV request and departure data for the testing period
    test_ev_request_dict = prepare_ev_request_data(parking_data_path, test_start_date, test_end_date)
    test_ev_departure_dict = prepare_ev_departure_data(parking_data_path, test_start_date, test_end_date) \
        if parking_version == '0' else prepare_ev_actual_departure_data(parking_data_path, test_start_date, test_end_date)

    test_episode_rewards = defaultdict(int)  # Record the rewards during the testing phase

    for episode in tqdm(range(1)):  # Typically, you run the test phase for a single episode
        s = test_env.reset()
        while test_env.timestamp <= test_env.end_time:  
            
            if test_env.timestamp.hour < 7 or test_env.timestamp.hour > 23:
                test_env.timestamp += timedelta(hours=1)
                continue

            # Add EVs to the environment, if there are EVs that have arrived at the current time
            current_requests = test_ev_request_dict.get(test_env.timestamp, [])  # Get the EVs that have arrived at the current time
            for ev in current_requests:
                test_env.add_ev(ev['requestID'], 
                                ev['arrival_time'], 
                                ev['departure_time'], 
                                ev['initial_soc'], 
                                ev['departure_soc'])
                test_env.current_parking_number += 1  # Increase the number of EVs in the environment
                        
            # Remove EVs that departed at the current time
            current_departures = test_ev_departure_dict.get(test_env.timestamp, [])
            for ev in current_departures:
                for agent_id, data in test_env.ev_data.items():
                    if ev['requestID'] == data['requestID']:
                        ev_departure_time = ev['departure_time'] if parking_version == '0' \
                            else ev['actual_departure_time']
                        test_env.remove_ev(agent_id, ev_departure_time, test_env.timestamp)
                        test_env.current_parking_number -= 1
            
            # Choose action based on the loaded actor network without adding noise
            a = agent.choose_action(s)
            s_, r, done, _ = test_env.step(a, test_env.timestamp)

            # Update the observation
            s = s_


    # Save the testing results
    test_soc_history_file = f'{result_dir}/test_soc_history.csv'
    test_charging_records_file = f'{result_dir}/test_charging_records.csv'
    test_env.soc_history.to_csv(test_soc_history_file, index=False)
    test_env.charging_records.to_csv(test_charging_records_file, index=False)

    # Save the load history during testing
    test_load_history_df = pd.DataFrame(test_env.load_history)
    test_load_history_file = f'{result_dir}/test_building_loading_history.csv'
    test_load_history_df.to_csv(test_load_history_file, index=False)

    print(f'Test results and histories saved to {result_dir}')