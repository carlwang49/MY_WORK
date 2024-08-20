import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
from EVBuildingEnv import EVBuildingEnv
from logger_config import configured_logger as logger
from utils.plot_results import plot_scores_epsilon
import torch
import torch.optim as optim
from utilities import prepare_ev_request_data, prepare_ev_departure_data, create_result_dir, set_seed
from ReplayBuffer import ReplayBuffer
from QNet import QNet   
from agent import train
from dotenv import load_dotenv
import os
load_dotenv()

NUMBER_OF_EPISODES = int(os.getenv('NUMBER_OF_EPISODES'))
LEARN_INTERVAL = int(os.getenv('LEARN_INTERVAL'))
RANDOM_STEPS = int(float(os.getenv('RANDOM_STEPS')))
TAU = float(os.getenv('TAU'))
GAMMA = float(os.getenv('GAMMA'))
AGENT_BUFFER_CAPACITY = int(float(os.getenv('AGENT_BUFFER_CAPACITY')))
AGENT_BATCH_SIZE = int(os.getenv('AGENT_BATCH_SIZE'))
LEARNING_RATE_ACTOR = float(os.getenv('LEARNING_RATE_ACTOR'))
LEARNING_RATE_CRITIC = float(os.getenv('LEARNING_RATE_CRITIC'))
EPSILON = float(os.getenv('EPSILON'))

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
dir_name = DIR_NAME = 'VDN-MARL'

# Define the path to the EV request data
parking_data_path = PARKING_DATA_PATH = f'../Dataset/Sim_Parking/ev_parking_data_from_2018-07-01_to_2018-12-31_{NUM_AGENTS}.csv'


def train_VDN_agent(env_name, lr, gamma, batch_size, buffer_limit, log_interval, max_episodes, max_epsilon,
                    min_epsilon,test_episodes, warm_up_steps, update_iter, chunk_size, update_target_interval,
                    recurrent):
    # set seed
    set_seed(42)
    
    # Define the start and end date of the EV request data
    ev_request_dict = prepare_ev_request_data(parking_data_path, start_date, end_date)
    ev_departure_dict = prepare_ev_departure_data(parking_data_path, start_date, end_date)

    # create env.
    env = EVBuildingEnv(num_agents, start_time, end_time)
    memory = ReplayBuffer(buffer_limit)

    # create networks
    q = QNet(env.observation_spaces, env.action_spaces, recurrent)
    q_target = QNet(env.observation_spaces, env.action_spaces, recurrent)
    q_target.load_state_dict(q.state_dict())
    optimizer = optim.Adam(q.parameters(), lr=lr)
    
    # For performance monitoring
    n_agents = len(env.observation_spaces)
    episode_rewards = [ [] for _ in range(n_agents)]
    epsilon_history = list()
    
    for episode_i in tqdm(range(max_episodes)):
        
        rewards_temp = [ [] for _ in range(n_agents)]
        epsilon = max(min_epsilon, max_epsilon - (max_epsilon - min_epsilon) * (episode_i / (0.6 * max_episodes)))
        state = env.reset()

        # done = [False for _ in range(env.n_agents)]
        with torch.no_grad():
            
            # initialize the hidden state
            hidden = q.init_hidden() 
            step_counter = 0
            
            # reset the timestamp to the start time of the environment
            env.timestamp = env.start_time 
            while env.timestamp <= env.end_time: 
                # skip the time
                if env.timestamp.hour < 7 or env.timestamp.hour > 23:
                    env.timestamp += timedelta(hours=1)
                    continue
                
                current_requests = ev_request_dict.get(env.timestamp, []) # get the EVs that have arrived at the current time
                
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
                    request_id = ev['requestID']
                    for agent_id, data in env.ev_data.items():
                        if data['requestID'] == request_id:
                            env.remove_ev(agent_id)
                            env.current_parking_number -= 1
                            break 
                
                step_counter += 1
                state = np.array(state)
                action, hidden = q.sample_action(torch.Tensor(state).unsqueeze(0), hidden, epsilon)
                action = action[0].data.cpu().numpy().tolist()
                next_state, reward, connected, info = env.step(action, env.timestamp)
                
                memory.put((state, action, (np.array(reward)).tolist(), next_state, [int(all(connected))]))
                state = next_state
        
                for i in range(n_agents):
                    rewards_temp[i].append(reward[i])
                
                if env.timestamp >= env.end_time: 
                    # log rewards
                    for i in range(n_agents):
                        episode_rewards[i].append(sum(rewards_temp[i]))
                    epsilon_history.append(epsilon)
                    break
            
        # train the agent        
        if memory.size() > warm_up_steps:
            train(q, q_target, memory, optimizer, gamma, batch_size, update_iter, chunk_size)

        if episode_i % update_target_interval:
            q_target.load_state_dict(q.state_dict())

        if (episode_i + 1) % log_interval == 0:
            rewards = 0
            for i in range(n_agents):
                rewards += sum(rewards_temp[i])
            logger.bind(console=True).info(f'Episode {episode_i} - Total reward: {rewards}')
            
    return q, episode_rewards, epsilon_history, env


if __name__ == '__main__':
    
    kwargs = {'env_name': 'dummy',
            'lr': LEARNING_RATE_ACTOR,
            'batch_size': AGENT_BATCH_SIZE,
            'gamma': GAMMA,
            'buffer_limit': AGENT_BUFFER_CAPACITY,
            'update_target_interval': LEARN_INTERVAL,
            'log_interval': LEARN_INTERVAL,
            'max_episodes': NUMBER_OF_EPISODES,
            'max_epsilon': 0.9,
            'min_epsilon': 0.25,
            'test_episodes': 5,
            'warm_up_steps': RANDOM_STEPS,
            'update_iter': 10,
            'chunk_size': 10,  # if not recurrent, internally, we use chunk_size of 1 and no gru cell is used.
            'recurrent': False}
    
    VDNagent, reward_history, epsilon_history, env = train_VDN_agent(**kwargs)
    result_dir = create_result_dir(f'{DIR_NAME}_alpha{ALPHA}_beta{BETA}_num{NUM_AGENTS}')
    plot_scores_epsilon(reward_history, epsilon_history, result_dir, moving_avg_window = 50)
    
    
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
    
    # Define the start and end date of the EV request data for testing
    test_ev_request_dict = prepare_ev_request_data(parking_data_path, test_start_date, test_end_date)
    test_ev_departure_dict = prepare_ev_departure_data(parking_data_path, test_start_date, test_end_date)

    # create test env.
    test_env = EVBuildingEnv(num_agents, test_start_time, test_end_time)

    # Reset the environment
    test_state = test_env.reset()

    # Load the trained model for testing
    model_save_path = os.path.join(result_dir, 'VDNagent_model.pth')
    torch.save(VDNagent.state_dict(), model_save_path)  # Save the trained model
    VDNagent.load_state_dict(torch.load(model_save_path))  # Load the model

    # Initialize variables for testing
    test_rewards = [ [] for _ in range(num_agents)]
    
    with torch.no_grad():
        hidden = VDNagent.init_hidden()
        test_env.timestamp = test_env.start_time

        while test_env.timestamp <= test_env.end_time: 
            
            if test_env.timestamp.hour < 7 or test_env.timestamp.hour > 23:
                test_env.timestamp += timedelta(hours=1)
                continue

            current_requests = test_ev_request_dict.get(test_env.timestamp, [])
            for ev in current_requests:
                test_env.add_ev(ev['requestID'], 
                                ev['arrival_time'], 
                                ev['departure_time'], 
                                ev['initial_soc'], 
                                ev['departure_soc'])
                test_env.current_parking_number += 1 
                        
            current_departures = test_ev_departure_dict.get(test_env.timestamp, [])
            # print(f"current_departures: {current_departures}")
            for ev in current_departures:
                request_id = ev['requestID']
                for agent_id, data in test_env.ev_data.items():
                    if data['requestID'] == request_id:
                        # print(f"EV {request_id} has departed at {test_env.timestamp}")
                        test_env.remove_ev(agent_id)
                        test_env.current_parking_number -= 1
                        break 
            
            test_state = np.array(test_state)
            action, hidden = VDNagent.sample_action(torch.Tensor(test_state).unsqueeze(0), hidden, epsilon=0)  # epsilon = 0 for testing
            action = action[0].data.cpu().numpy().tolist()
            next_test_state, reward, connected, info = test_env.step(action, test_env.timestamp)
            test_state = next_test_state

            for i in range(num_agents):
                test_rewards[i].append(reward[i])
            
    
    # Save test soc history and charging records
    test_soc_history_file = f'{result_dir}/test_soc_history.csv'
    test_charging_records_file = f'{result_dir}/test_charging_records.csv'
    test_env.soc_history.to_csv(test_soc_history_file, index=False)
    test_env.charging_records.to_csv(test_charging_records_file, index=False)
    
    # Save test load history
    test_load_history_df = pd.DataFrame(test_env.load_history)
    test_load_history_file = f'{result_dir}/test_building_loading_history.csv'
    test_load_history_df.to_csv(test_load_history_file, index=False)

    print(f'Test results and histories saved to {result_dir}')