from EVChargingEnv import EVChargingEnv
import numpy as np
from datetime import datetime, timedelta
from logger_config import configured_logger as logger
from utils import prepare_ev_request_data, prepare_ev_departure_data, create_result_dir

# TODO: 合併 start_time, start_date, end_time, end_date

# Define the start and end date of the EV request data
start_date = START_DATE = '2018-07-01'
end_date = END_DATE = '2018-07-02'

# Define the start and end time of the EV request data
start_time = START_TIME = datetime(2018, 7, 1)
end_time = END_TIME = datetime(2018, 7, 2)

num_agents = NUM_AGENTS = 10
parking_data_path = PARKING_DATA_PATH = '../Dataset/Sim_Parking/ev_parking_data_from_2018-07-01_to_2018-12-31.csv'


if __name__ == '__main__':
    """
    Run a simulation of the EV charging environment with random EV request data and random charging power selection.
    """
    # Create a directory for storing results of the simulation
    result_dir = create_result_dir('RandomPower')
    
    # Define the start and end date of the EV request data
    ev_request_dict = prepare_ev_request_data(parking_data_path, start_date, end_date)
    ev_departure_dict = prepare_ev_departure_data(parking_data_path, start_date, end_date)
    
    # Set random seed for reproducibility and create an EV charging environment
    env = EVChargingEnv(num_agents, start_time, end_time)
    current_time = start_time  
    while current_time <= end_time:
        
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
                env.remove_ev(agent_idx)
                env.current_parking_number -= 1
                
        # Update the SoC of each connected EV
        for agent_idx, is_parking in enumerate(env.current_parking):
            if is_parking:
                ev_data = env.get_agent_status(agent_idx)
                SoC_lower_bound, SoC_upper_bound = env.get_soc_max_and_min(agent_idx, current_time + timedelta(hours=1))
                P_max_tk, P_min_tk = env.get_P_max_tk_and_P_min_tk(agent_idx, SoC_lower_bound, SoC_upper_bound)
                action = np.random.uniform(P_min_tk, P_max_tk)  # Randomly select a charging power
                
                # Update the SoC of the connected EV
                env.step(agent_idx, action, current_time, SoC_lower_bound, SoC_upper_bound, time_interval=60)
        
        current_time += timedelta(hours=1)

    # Save the charging records and SoC history to CSV files
    env.soc_history.to_csv(f'../Result/{result_dir}/soc_history.csv', index=False)
    env.charging_records.to_csv(f'../Result/{result_dir}/charging_records.csv', index=False)