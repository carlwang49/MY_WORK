from tqdm import tqdm
import pandas as pd
import sys
from datetime import datetime
from utils.plot_results import plot_scores_epsilon
from QLearningAgent import QLearningAgent
from utilities import prepare_ev_request_data, prepare_ev_departure_data, create_result_dir
from EVBuildingEnv import EVBuildingEnv

# Import the environment and plotting functions
sys.path.insert(0,'./utils')

# Define the start and end date of the EV request data
start_date = START_DATE = '2018-07-01'
end_date = END_DATE = '2018-12-31'

# Define the start and end time of the EV request data
start_time = START_TIME = datetime(2018, 7, 1)
end_time = END_TIME = datetime(2018, 12, 31)

# Define the number of agents
num_agents = NUM_AGENTS = 10
parking_data_path = PARKING_DATA_PATH = '../Dataset/Sim_Parking/ev_parking_data_from_2018-07-01_to_2018-12-31.csv'


# HYPERPARAMETERS
N_AGENTS = 10
NUM_EPISODES = 6000
EPS_DECAY = 0.9999
EPS_MIN = 0.3
STEP_SIZE = 0.1
GAMMA = 0.99
MAX_STEPS_DONE = 70


def train_QL_agents(n_agents, num_episodes, max_steps_done, eps_decay, eps_min, step_size, gamma):

    # Define the start and end date of the EV request data
    ev_request_dict = prepare_ev_request_data(parking_data_path, start_date, end_date)
    ev_departure_dict = prepare_ev_departure_data(parking_data_path, start_date, end_date)

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
                    request_id = ev['requestID']
                    for agent_id, data in env.ev_data.items():
                        if data['requestID'] == request_id:
                            env.remove_ev(agent_id)
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
    agents, reward_history, epsilon_history, env = train_QL_agents(N_AGENTS, NUM_EPISODES, MAX_STEPS_DONE, EPS_DECAY,
                                                          EPS_MIN, STEP_SIZE, GAMMA)
    
    result_dir = create_result_dir('IQL')
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