from tqdm import tqdm
import pandas as pd
from datetime import datetime, timedelta
from utils.plot_results import plot_scores_epsilon
from QLearningAgent import QLearningAgent
from utilities import prepare_ev_request_data, prepare_ev_departure_data, prepare_ev_actual_departure_data, create_result_dir, set_seed
from EVBuildingEnv import EVBuildingEnv
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
dir_name = DIR_NAME = 'IQL-MARL'
random_seed = RANDOM_SEED = int(os.getenv('RANDOM_SEED'))

# Define the path to the EV request data
parking_version = PARKING_VERSION = os.getenv('PARKING_VERSION')
parking_data_path = PARKING_DATA_PATH = f'../Dataset/Sim_Parking/ev_parking_data_v{PARKING_VERSION}_from_2018-07-01_to_2018-12-31_{NUM_AGENTS}.csv'

# HYPERPARAMETERS
N_AGENTS = NUM_AGENTS
NUM_EPISODES = int(os.getenv('NUMBER_OF_EPISODES'))
EPS_DECAY = 0.9999
EPS_MIN = 0.3
STEP_SIZE = 0.1
GAMMA = float(os.getenv('GAMMA'))


def train_QL_agents(n_agents, num_episodes, eps_decay, eps_min, step_size, gamma):
    
    # set seed
    set_seed(RANDOM_SEED)

    # Define the start and end date of the EV request data
    ev_request_dict = prepare_ev_request_data(parking_data_path, start_date, end_date)
    ev_departure_dict = prepare_ev_departure_data(parking_data_path, start_date, end_date) \
        if parking_version == '0' else prepare_ev_actual_departure_data(parking_data_path, start_date, end_date)

    # create env.
    env = EVBuildingEnv(num_agents, start_time, end_time)

    # get numer of states and actions from environment
    n_states = 3000000
    n_actions = env.action_spaces[0].n

    # Initialise agents
    agents = []
    for _ in range(n_agents):
        QL_agent = QLearningAgent(num_actions=n_actions, num_states=n_states,
                      eps_start = 1.0, eps_decay=eps_decay, eps_min=eps_min,
                      step_size=step_size, gamma=gamma)
        agents.append(QL_agent)

    # Monitor the scores and epsilon for each episode
    episode_rewards = [ [] for _ in range(n_agents)]
    epsilon_history = list()
    
    # for episode in num_episodes
    for episode in tqdm(range(num_episodes)):
        
        rewards_temp = [ [] for _ in range(n_agents)]
        
        # get initial state and actions
        states = env.reset()
        for i in range(len(agents)):
            action = agents[i].agent_start(states[i])
        
        # reset the timestamp to the start time of the environment
        env.timestamp = env.start_time 
        
        rewards = [0 for _ in range(n_agents)]
        steps = 0
        
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
                request_id = ev['requestID']
                for agent_id, data in env.ev_data.items():
                    if data['requestID'] == request_id:
                        ev_departure_time = ev['departure_time'] if parking_version == '0' \
                            else ev['actual_departure_time']
                        env.remove_ev(agent_id, ev_departure_time, env.timestamp)
                        env.current_parking_number -= 1
                        break 
                    
            steps += 1
            actions = []
            for i in range(n_agents):
                action = agents[i].agent_step(rewards[i], states[i])
                actions.append(action)
            
            next_states, rewards, dones, info = env.step(actions, env.timestamp)

            for i in range(n_agents):
                # Update q values
                rewards_temp[i].append(rewards[i])
            
            for i in range(n_agents):
                # Update q values
                agents[i].agent_end(states[i], rewards[i]) #update q values last time
            
            if episode%1000==0:
                for i in range(n_agents):
                    episode_rewards[i].append(sum(rewards_temp[i]))
            epsilon_history.append(agents[0].epsilon)
            
            if env.timestamp >= env.end_time: 
                if episode%1000==0:
                    for i in range(n_agents):
                        episode_rewards[i].append(sum(rewards_temp[i]))
                epsilon_history.append(agents[0].epsilon)
                break
                
            states = next_states
    return agents, episode_rewards, epsilon_history, env


if __name__ == '__main__':
    agents, reward_history, epsilon_history, env = train_QL_agents(N_AGENTS, NUM_EPISODES, EPS_DECAY,
                                                          EPS_MIN, STEP_SIZE, GAMMA)
    
    result_dir = create_result_dir(f'{DIR_NAME}_{start_date_without_year}_{end_date_without_year}_{NUM_AGENTS}_sim_v{PARKING_VERSION}')
    plot_scores_epsilon(reward_history, epsilon_history, result_dir, moving_avg_window = 50)
    
    # Save the Q-learning agents
    for i, agent in enumerate(agents):
        agent_file = os.path.join(result_dir, f'agent_{i}.pkl')
        agent.save(agent_file)
    
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
    test_ev_departure_dict = prepare_ev_departure_data(parking_data_path, test_start_date, test_end_date) \
        if parking_version == '0' else prepare_ev_actual_departure_data(parking_data_path, test_start_date, test_end_date)

    # create test env.
    test_env = EVBuildingEnv(num_agents, test_start_time, test_end_time)

    # Load the trained agents for testing
    for i, agent in enumerate(agents):
        agent_file = os.path.join(result_dir, f'agent_{i}.pkl')
        agent.load(agent_file)

    # Initialize variables for testing
    test_states = test_env.reset()
    
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
        for ev in current_departures:
            request_id = ev['requestID']
            for agent_id, data in test_env.ev_data.items():
                if data['requestID'] == request_id:
                    ev_departure_time = ev['departure_time'] if parking_version == '0' \
                            else ev['actual_departure_time']
                    test_env.remove_ev(agent_id, ev_departure_time, test_env.timestamp)
                    test_env.current_parking_number -= 1
                    break 

        test_actions = [agent.test_step(test_states[i]) for i, agent in enumerate(agents)]
        next_test_states, test_rewards_step, _, _ = test_env.step(test_actions, test_env.timestamp)

        test_states = next_test_states
    
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