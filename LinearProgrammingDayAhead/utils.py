import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

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


def get_num_of_evs_in_each_hours(day_ahead_schedule_df: pd.DataFrame, current_time: datetime):
    
    day_ahead_schedule_df = day_ahead_schedule_df[(day_ahead_schedule_df['date'] == current_time.date())].copy()
    number_of_ev_in_hours = []
    current_ev_num = 0
    
    for hour in range(24):
        current_ev_num += len(day_ahead_schedule_df[(day_ahead_schedule_df['arrival_time'].dt.hour == hour)])
        current_ev_num -= len(day_ahead_schedule_df[(day_ahead_schedule_df['departure_time'].dt.hour == hour)])
        number_of_ev_in_hours.append(current_ev_num)   
    
    return number_of_ev_in_hours


def get_tou_price(current_time: datetime):
    
    tou_price = []
    if current_time.weekday() < 5:
        tou_price = [0.056] * 8 + [0.092] * 4 + [0.267] * 6 + [0.092] * 5 + [0.056] * 1
    else:
        tou_price = [0.056] * 24      
    return tou_price

def get_rtp_price(start_time, end_time):
    # 读取实时价格数据
    real_time_price = pd.read_csv('../Dataset/RTP/electricity_prices_from_201807010000_to_201812312359.csv')
    real_time_price['datetime'] = pd.to_datetime(real_time_price['datetime'])
    
    # 筛选所需时间范围的数据
    real_time_price = real_time_price[(real_time_price['datetime'] >= start_time) & (real_time_price['datetime'] < end_time + timedelta(hours=1))].copy()
    real_time_price.sort_values(by='datetime', inplace=True)
    
    # 创建完整的时间序列索引
    full_time_index = pd.date_range(start=start_time, end=end_time, freq='H')
    
    # 将数据框设置为时间索引
    real_time_price.set_index('datetime', inplace=True)
    
    # 使用完整的时间索引重新索引数据框，并使用前向填充方法填充缺失值
    real_time_price = real_time_price.reindex(full_time_index).ffill().reset_index()
    
    # 重命名列
    real_time_price.rename(columns={'index': 'datetime'}, inplace=True)
    
    return real_time_price