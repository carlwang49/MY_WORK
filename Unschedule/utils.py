import pandas as pd
import numpy as np
import os
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
    total_files = len([file for file in os.listdir(env_dir)])
    result_dir = os.path.join(env_dir, f'{total_files + 1}')
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)