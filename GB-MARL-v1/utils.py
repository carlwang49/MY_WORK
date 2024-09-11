import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import pickle
import random
import torch

def prepare_ev_request_data(parking_data_path, start_date, end_date):
    """
    Prepare electric vehicle (EV) request data.

    Args:
        parking_data_path (str): The file path of the parking data.
        start_date (str): The start date for filtering the data.
        end_date (str): The end date for filtering the data.

    Returns:
        dict: A dictionary containing EV request data grouped by arrival time.
    """
    ev_request_data = pd.read_csv(parking_data_path, parse_dates=['arrival_time', 'departure_time']) 
    ev_request_data = ev_request_data[(ev_request_data['date'] >= start_date) & (ev_request_data['date'] < end_date)].copy()
    ev_request_data['date'] = pd.to_datetime(ev_request_data['date']).dt.date 
    ev_request_dict = ev_request_data.groupby(ev_request_data['arrival_time']).apply(lambda x: x.to_dict(orient='records')).to_dict()
    
    return ev_request_dict



def prepare_ev_departure_data(parking_data_path, start_date, end_date):
    """
    Prepare electric vehicle (EV) request(departure) data.

    Args:
        parking_data_path (str): The file path of the parking data.
        start_date (str): The start date for filtering the data.
        end_date (str): The end date for filtering the data.

    Returns:
        dict: A dictionary containing EV request data grouped by departure data.
    """
    ev_request_data = pd.read_csv(parking_data_path, parse_dates=['arrival_time', 'departure_time']) 
    ev_request_data = ev_request_data[(ev_request_data['date'] >= start_date) & (ev_request_data['date'] < end_date)].copy()
    ev_request_data['date'] = pd.to_datetime(ev_request_data['date']).dt.date 
    ev_departure_dict = ev_request_data.groupby(ev_request_data['departure_time']).apply(lambda x: x.to_dict(orient='records')).to_dict()
    
    return ev_departure_dict


def prepare_ev_actual_departure_data(parking_data_path, start_date, end_date):
    """
    Prepare electric vehicle (EV) actual departure data.
    
    Args:
        parking_data_path (str): The file path of the parking data.
        start_date (str): The start date for filtering the data.
        end_date (str): The end date for filtering the data.
        
    Returns:
        dict: A dictionary containing EV actual departure data grouped by departure data.
    """
    
    ev_request_data = pd.read_csv(parking_data_path, parse_dates=['arrival_time', 'departure_time', 'actual_departure_time'])   
    ev_request_data = ev_request_data[(ev_request_data['date'] >= start_date) & (ev_request_data['date'] < end_date)].copy()
    ev_request_data['date'] = pd.to_datetime(ev_request_data['date']).dt.date 
    ev_actual_departure_dict = ev_request_data.groupby(ev_request_data['actual_departure_time']).apply(lambda x: x.to_dict(orient='records')).to_dict()
    
    return ev_actual_departure_dict


def setup_logger(filename):
    """Set up the logger"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(filename, mode='w')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s--%(levelname)s--%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger

#### Decrepated ####
# def create_result_dir(method_name='EVBuildingEnv'):
#     """
#     Create a directory for storing results of a method.

#     Args:
#         method_name (str, optional): The name of the method. Defaults to 'EVBuildingEnv'.

#     Returns:
#         str: The path of the created result directory.
#     """
#     env_dir = os.path.join('../Result', method_name)
    
#     if not os.path.exists(env_dir):
#         os.makedirs(env_dir)
#     total_files = len([file for file in os.listdir(env_dir)])
#     result_dir = os.path.join(env_dir, f'{total_files + 1}')
#     os.makedirs(result_dir)
    
#     return result_dir

def create_result_dir(method_name='EVBuildingEnv'):
    """
    Create a directory for storing results of a method.

    Args:
        method_name (str, optional): The name of the method. Defaults to 'EVBuildingEnv'.

    Returns:
        str: The path of the created result directory.
    """
    env_dir = os.path.join('../Result', method_name)
    
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    
    # get the maximum directory number
    existing_dirs = [int(file) for file in os.listdir(env_dir) if file.isdigit()]
    
    if existing_dirs:
        max_dir_num = max(existing_dirs)
    else:
        max_dir_num = 0
    
    result_dir = os.path.join(env_dir, f'{max_dir_num + 1}')
    os.makedirs(result_dir)
    
    return result_dir


def get_running_reward(arr: np.ndarray, window=100):
        """calculate the running reward, i.e. average of last `window` elements from rewards"""
        running_reward = np.zeros_like(arr)
        effective_window = min(window, len(arr))  # ensure the window is not greater than the length of rewards
        for i in range(effective_window - 1):
            running_reward[i] = np.mean(arr[:i + 1])
        for i in range(effective_window - 1, len(arr)):
            running_reward[i] = np.mean(arr[i - effective_window + 1:i + 1])
        return running_reward
    

def min_max_scaling(value, min_val, max_val):
    return round((value - min_val) / (max_val - min_val), 2)


def standardize(value, mean, std):
    return (value - mean) / std


def plot_training_results(episode_rewards, args, result_dir):
    # Calculate the average reward per episode
    avg_rewards = np.mean(list(episode_rewards.values()), axis=0)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(1, args.episode_num + 1)

    # Plot the average reward
    ax.plot(x, avg_rewards, label='Average Reward', color='blue', linewidth=2)
    
    # Plot the running average reward
    running_avg_rewards = get_running_reward(avg_rewards)
    ax.plot(x, running_avg_rewards, label='Running Average Reward', color='orange', linestyle='--', linewidth=2)

    # Customize the plot
    ax.legend(fontsize=12)
    ax.set_xlabel('Episode', fontsize=14)
    ax.set_ylabel('Reward', fontsize=14)
    ax.set_title('Training Agent Result of GB-MARL', fontsize=16)
    ax.grid(True)
    
    # Save the figure
    plt.savefig(os.path.join(result_dir, 'training_result_agent_reward_GB-MARL.png'))
    plt.show()
    

def plot_global_training_results(episode_global_rewards, args, result_dir):
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(1, args.episode_num + 1)

    # Plot the global reward
    ax.plot(x, episode_global_rewards, label='Global Reward', color='green', linewidth=2, linestyle=':')
    
    # Customize the plot
    ax.legend(fontsize=12)
    ax.set_xlabel('Episode', fontsize=14)
    ax.set_ylabel('Reward', fontsize=14)
    ax.set_title('Training Global Result of GB-MARL', fontsize=16)
    ax.grid(True)
    
    # Save the figure
    plt.savefig(os.path.join(result_dir, 'training_result_global_reward_GB-MARL.png'))
    plt.show()
    
    # Save the global rewards to a pickle file
    with open(os.path.join(result_dir, 'global_reward.pkl'), 'wb') as f:
        pickle.dump(episode_global_rewards, f)
        
        
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)