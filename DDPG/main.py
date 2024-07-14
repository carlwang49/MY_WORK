import torch
import numpy as np
from logger_config import configured_logger as logger
from EVBuildingEnv import EVBuildingEnv
from datetime import datetime, timedelta
from utils import prepare_ev_request_data, prepare_ev_departure_data, create_result_dir, plot_training_results
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from DDPG import DDPG
from NetWork import ReplayBuffer
import pickle
from dotenv import load_dotenv
import os
load_dotenv()

# Define the start and end datetime of the EV request data
start_datetime = datetime(2018, 7, 1)
end_datetime = datetime(2018, 10, 1)

# Define the start and end date of the EV request data
start_date = START_DATE = str(start_datetime.date())
end_date = END_DATE = str(end_datetime.date())

# Define the start and end time of the EV request data
start_time = START_TIME = start_datetime
end_time = END_TIME = end_datetime

# Define the number of agents
num_agents = NUM_AGENTS = int(os.getenv('NUM_AGENTS'))
parking_data_path = PARKING_DATA_PATH = f'../Dataset/Sim_Parking/ev_parking_data_from_2018-07-01_to_2018-12-31_{num_agents}.csv'

# Define hyperparameters
random_steps = RANDOM_STEPS  = 25e3  # Take the random actions in the beginning for the better exploration
update_freq = UPDATE_FREQ = 50  # Take 50 steps,then update the networks 50 times
episode_rewards = EPISODE_REWARDS = defaultdict(int)  # Record the rewards during the evaluating
episode_num = EPISODE_NUM = 3000

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
    
    
    for episode in tqdm(range(episode_num)):
        episode_steps = 0
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
                        env.remove_ev(agent_id)
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

    # Save the models
    torch.save(agent.actor.state_dict(), os.path.join(result_dir, 'actor.pth'))
    torch.save(agent.critic.state_dict(), os.path.join(result_dir, 'critic.pth'))
    
    
    # plot_training_results
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