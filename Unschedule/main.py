from EVChargingEnv import EVChargingEnv
import numpy as np
from datetime import datetime, timedelta
from logger_config import configured_logger as logger
from utils import prepare_ev_request_data, prepare_ev_departure_data, prepare_ev_actual_departure_data, create_result_dir, set_seed
import pandas as pd
from dotenv import load_dotenv
import os
load_dotenv()

start_datetime_str = os.getenv('TEST_START_DATETIME', '2018-10-01')
end_datetime_str = os.getenv('TEST_END_DATETIME', '2018-10-07')

# Define the start and end datetime of the EV request data
start_datetime = datetime.strptime(start_datetime_str, '%Y-%m-%d')
end_datetime = datetime.strptime(end_datetime_str, '%Y-%m-%d')

# Define the start and end date of the EV request data
start_date = START_DATE = str(start_datetime.date())
end_date = END_DATE = str(end_datetime.date())
end_date_minus_one = END_DATE_MINUS_ONE = str((datetime.strptime(END_DATE, '%Y-%m-%d') - timedelta(days=1)).date())

start_date_without_year = START_DATE[5:]  # Assuming the format is 'YYYY-MM-DD'
end_date_without_year = END_DATE_MINUS_ONE[5:]

# Define the start and end time of the EV request data
start_time = START_TIME = start_datetime
end_time = END_TIME = end_datetime

# Define the number of agents
num_agents = NUM_AGENTS = int(os.getenv('NUM_AGENTS'))

# Define the path to the EV request data
parking_version = PARKING_VERSION = os.getenv('PARKING_VERSION')
parking_data_path = PARKING_DATA_PATH = f'../Dataset/Sim_Parking/ev_parking_data_v{PARKING_VERSION}_from_2018-07-01_to_2018-12-31_{NUM_AGENTS}.csv'

# Define the directory name to save the result
# dir_name = DIR_NAME = 'GB-MARL-Discrete'
dir_name = DIR_NAME = 'Unschedule'


if __name__ == '__main__':
    """
    Run a simulation of the EV charging environment with random EV request data and random charging power selection.
    """
    # Set random seed for reproducibility
    set_seed(30)
    
    # Create a directory for storing results of the simulation
    result_dir = create_result_dir(f'{DIR_NAME}_{start_date_without_year}_{end_date_without_year}_{NUM_AGENTS}_sim_v{PARKING_VERSION}')
    
    # Define the start and end date of the EV request data
    ev_request_dict = prepare_ev_request_data(parking_data_path, start_date, end_date)
    ev_departure_dict = prepare_ev_departure_data(parking_data_path, start_date, end_date) \
        if parking_version == '0' else prepare_ev_actual_departure_data(parking_data_path, start_date, end_date)
    
    # Set random seed for reproducibility and create an EV charging environment
    env = EVChargingEnv(num_agents, start_time, end_time)
    current_time = start_time  
    while current_time <= end_time:
        
        if current_time.hour < 7 or current_time.hour > 23:
            current_time += timedelta(hours=1)
            continue
            
        current_requests = ev_request_dict.get(current_time, [])
        # Add EVs that arrived at the current time
        if current_requests:
            for ev in current_requests:
                env.add_ev(ev['requestID'], 
                            ev['arrival_time'], 
                            ev['departure_time'], 
                            ev['initial_soc'], 
                            ev['departure_soc'])
                
                env.current_parking_number += 1 # increase the number of EVs in the environment

        current_departures = ev_departure_dict.get(current_time, [])
        # Remove EVs that departed at the current time
        if current_departures:
            for ev in current_departures:
                agent_idx = np.where([ev['requestID'] == data['requestID'] for data in env.ev_data])[0][0]
                ev_departure_time = ev['departure_time'] if parking_version == '0' \
                            else ev['actual_departure_time']
                env.remove_ev(agent_idx, ev_departure_time, env.timestamp)
                env.current_parking_number -= 1
                
        # Update the SoC of each connected EV
        total_action = 0
        for agent_idx, is_parking in enumerate(env.current_parking):
            if is_parking:
                ev_data = env.get_agent_status(agent_idx)
                SoC_lower_bound, SoC_upper_bound = env.get_soc_max_and_min(agent_idx, current_time + timedelta(hours=1))
                P_max_tk, P_min_tk = env.get_P_max_tk_and_P_min_tk(agent_idx, SoC_lower_bound, SoC_upper_bound)
                action = P_max_tk  # Randomly select a charging power
                total_action += action
                # Update the SoC of the connected EV
                env.step(agent_idx, action, current_time, SoC_lower_bound, SoC_upper_bound, time_interval=60)
        
        # Update the building load
        env.building_load.loc[env.building_load['Date'] == current_time, 'Total_Power(kWh)'] += total_action
        env.load_history.append({
                'current_time': current_time,
                'original_load': env.original_load,
                'total_load': env.building_load.loc[env.building_load['Date'] == current_time, 'Total_Power(kWh)'].values[0],
                'total_action_impact': total_action  
            })
        current_time += timedelta(hours=1)

    # Save the charging records and SoC history to CSV files
    env.soc_history.to_csv(f'../Result/{result_dir}/soc_history.csv', index=False)
    env.charging_records.to_csv(f'../Result/{result_dir}/charging_records.csv', index=False)
    
    # save load history
    load_history_df = pd.DataFrame(env.load_history)
    load_history_file = f'{result_dir}/building_loading_history.csv'
    load_history_df.to_csv(load_history_file, index=False)
    
    print(f'Test results and histories saved to {result_dir}')