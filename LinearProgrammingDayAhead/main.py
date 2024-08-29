import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
from scipy.optimize import linprog
from logger_config import configured_logger as logger
from utils import create_result_dir, get_num_of_evs_in_each_hours, get_rtp_price, set_building_time_range
from dotenv import load_dotenv
import os
load_dotenv()

# Contract capacity and capacity price
contract_capacity = CONTRACT_CAPACITY = int(os.getenv('CONTRACT_CAPACITY')) 
capacity_price = CAPACITY_PRICE = float(os.getenv('CAPACITY_PRICE'))
weight = 1e6  # Adjusted weight for capacity constraint penalty

# Dates for EV request data
start_date = START_DATE = os.getenv('TEST_START_DATETIME', '2018-07-01')
end_date = END_DATE = os.getenv('TEST_END_DATETIME', '2018-10-01')
end_date_minus_one = END_DATE_MINUS_ONE = str((datetime.strptime(END_DATE, '%Y-%m-%d') - timedelta(days=1)).date())

# Extract year-agnostic dates
start_date_without_year = START_DATE[5:]  # Assuming format is 'YYYY-MM-DD'
end_date_without_year = END_DATE_MINUS_ONE[5:]

# Define the start and end datetime of the EV request data
start_time = START_TIME = datetime.strptime(start_date, '%Y-%m-%d')
end_time = END_TIME = datetime.strptime(end_date, '%Y-%m-%d')

# Number of agents and file paths
num_agents = NUM_AGENTS = int(os.getenv('NUM_AGENTS'))
parking_data_path = PARKING_DATA_PATH = f'../Dataset/Sim_Parking/ev_parking_data_from_2018-07-01_to_2018-12-31_{NUM_AGENTS}.csv'
building_load_file = BUILDING_LOAD_FILE = '../Dataset/BuildingEnergyLoad/BuildingConsumptionLoad.csv'

# Environment settings
min_soc = MIN_SOC = float(os.getenv('SOC_MIN', 0.2))  
max_soc = MAX_SOC = float(os.getenv('SOC_MAX', 0.8))  
c_k = C_K = float(os.getenv('BATTERY_CAPACITY', 60))  
eta = ETA = float(os.getenv('CHARGING_EFFICIENCY', 0.95))  
max_charging_power = MAX_CHARGING_POWER = int(os.getenv('MAX_CHARGING_POWER', 150))  
max_discharging_power = MAX_DISCHARGING_POWER = int(os.getenv('MAX_DISCHARGING_POWER', -150))  

if __name__ == '__main__':
    # Initialize dataframes
    charging_records = pd.DataFrame(columns=['requestID', 'arrival_time', 'departure_time', 'initial_soc', 'departure_soc', 'final_soc', 'charging_power', 'charging_time']) 
    soc_history = pd.DataFrame(columns=['requestID', 'current_time', 'soc', 'current_energy', 'charging_power/discharging_power', 'cost', 'total_cost']) 
    
    # Load EV request data
    day_ahead_schedule_df = pd.read_csv(parking_data_path)
    day_ahead_schedule_df['date'] = pd.to_datetime(day_ahead_schedule_df['date']).dt.date
    day_ahead_schedule_df['arrival_time'] = pd.to_datetime(day_ahead_schedule_df['arrival_time'])
    day_ahead_schedule_df['departure_time'] = pd.to_datetime(day_ahead_schedule_df['departure_time'])
    
    # Initialize building load
    building_load = pd.read_csv(building_load_file, parse_dates=['Date'])
    building_load = set_building_time_range(building_load, start_time, end_time)
    load_history = pd.DataFrame(columns=['current_time', 'original_load', 'total_load', 'total_action_impact'])
    total_action_impact = defaultdict(float)
    
    # Create results directory
    result_dir = create_result_dir(f'DayAheadSchedule_{start_date_without_year}_{end_date_without_year}_num{NUM_AGENTS}')
    
    # Get real-time price data
    real_time_price = get_rtp_price(start_time, end_time) 
    
    current_time = start_time  
    while current_time < end_time:
        number_of_ev_in_hours = get_num_of_evs_in_each_hours(day_ahead_schedule_df, current_time)
        
        # Get EV requests for the current day
        ev_request_list = day_ahead_schedule_df[day_ahead_schedule_df['date'] == current_time.date()].to_dict(orient='records') 

        for ev_request in ev_request_list:
            
            # Extract EV request information
            initial_soc, final_soc = ev_request['initial_soc'], ev_request['departure_soc']
            arrival_time, departure_time = ev_request['arrival_time'].hour, ev_request['departure_time'].hour
            num_intervals = departure_time - arrival_time
            
            # Calculate charging/discharging power constraints
            max_powers = [max_charging_power / number_of_ev_in_hours[hour] for hour in range(arrival_time, departure_time)]
            min_powers = [max_discharging_power / number_of_ev_in_hours[hour] for hour in range(arrival_time, departure_time)]
            
            # Calculate energy requirements
            initial_energy = initial_soc * c_k
            required_energy = final_soc * c_k
            min_energy = min_soc * c_k
            max_energy = max_soc * c_k

            # Define time intervals and price data
            time_intervals = list(range(arrival_time, departure_time))
            rtp_price = real_time_price[(real_time_price['datetime'] >= ev_request['arrival_time']) & (real_time_price['datetime'] < ev_request['departure_time'])]
            relevant_prices = rtp_price['average_price'].values

            # building load data
            building_current_load = building_load[(building_load['Date'] >= ev_request['arrival_time']) & 
                                      (building_load['Date'] < ev_request['departure_time'])]['Total_Power(kWh)'].values
            total_load_cost = building_current_load * relevant_prices

            # Linear programming problem definition
            c = np.concatenate((total_load_cost + relevant_prices, total_load_cost - relevant_prices, weight * np.array([capacity_price] * num_intervals)))
 
            # Ensure total energy change equals required energy change
            A_eq = np.concatenate((np.ones(num_intervals), -np.ones(num_intervals), np.zeros(num_intervals))).reshape(1, -1)
            b_eq = np.array([required_energy - initial_energy])
            
            # Charge/discharge power constraints
            A_ub = np.zeros((5 * num_intervals, 3 * num_intervals))  
            b_ub = np.zeros(5 * num_intervals)
            
            for i in range(num_intervals):
                A_ub[i, i] = 1 
                A_ub[num_intervals + i, num_intervals + i] = 1  
                b_ub[i] = max_powers[i]
                b_ub[num_intervals + i] = -min_powers[i]
            
            # SoC constraints
            for i in range(num_intervals):
                A_ub[2 * num_intervals + i, :i + 1] = 1
                A_ub[2 * num_intervals + i, num_intervals:num_intervals + i + 1] = -1
                b_ub[2 * num_intervals + i] = max_energy - initial_energy
                
                A_ub[3 * num_intervals + i, :i + 1] = -1
                A_ub[3 * num_intervals + i, num_intervals:num_intervals + i + 1] = 1
                b_ub[3 * num_intervals + i] = initial_energy - min_energy

            # Contract capacity constraints with penalty variables
            for i in range(num_intervals):
                A_ub[4 * num_intervals + i, i] = 1
                A_ub[4 * num_intervals + i, num_intervals + i] = -1
                A_ub[4 * num_intervals + i, 2 * num_intervals + i] = -1  # penalty variable
                b_ub[4 * num_intervals + i] = \
                    contract_capacity - building_load.loc[building_load['Date'] == current_time + timedelta(hours=time_intervals[i]), 'Total_Power(kWh)'].values[0]

            # Variable bounds
            bounds = [(0, max_powers[i]) for i in range(num_intervals)] + [(0, -min_powers[i]) for i in range(num_intervals)] + [(0, None) for i in range(num_intervals)]
        
            # Solve the linear programming problem
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method='highs')
            
            if res.success:
                current_energy = initial_energy
                total_cost = 0
                current_soc = initial_soc
                for i in range(num_intervals):
                    action, energy, cost = 'none', 0, 0
                    if res.x[i] > 0:
                        action = 'charge'
                        energy = res.x[i]
                        cost = res.x[i] * relevant_prices[i]
                    elif res.x[num_intervals + i] > 0:
                        action = 'discharge'
                        energy = res.x[num_intervals + i]
                        cost = -res.x[num_intervals + i] * relevant_prices[i]
                
                    current_energy += energy if action == 'charge' else -energy
                    current_soc = current_energy / c_k
                    total_cost += cost

                    soc_history.loc[len(soc_history)] = ({
                        'requestID': ev_request['requestID'],    
                        'current_time': current_time + timedelta(hours=time_intervals[i]),
                        'soc': initial_soc if time_intervals[i] == time_intervals[0] else current_soc,    
                        'current_energy': current_energy,
                        'charging_power/discharging_power': -energy if action == 'discharge' else energy,
                        'cost': cost,
                        'total_cost': total_cost
                    })
                    total_action_impact[current_time + timedelta(hours=time_intervals[i])] += energy if action == 'charge' else -energy
                    
            else:
                raise ValueError("Charging demand cannot be met within the given time frame.")
                
            charging_records.loc[len(charging_records)] = ({
                        'requestID': ev_request['requestID'], 
                        'arrival_time': ev_request['arrival_time'], 
                        'departure_time': ev_request['departure_time'], 
                        'initial_soc': initial_soc, 
                        'departure_soc': final_soc,
                        'final_soc': final_soc, 
                        'charging_power': (final_soc - initial_soc) * c_k, 
                        'charging_time': num_intervals
                    })
        current_time += timedelta(days=1)

    # Save results
    soc_history.to_csv(f'../Result/{result_dir}/soc_history.csv', index=False)
    charging_records.to_csv(f'../Result/{result_dir}/charging_records.csv', index=False)

    # Calculate load history
    current_time = start_time
    while current_time <= end_time:
        original_load = building_load.loc[building_load['Date'] == current_time, 'Total_Power(kWh)'].values[0]
        if total_action_impact.get(current_time):
            load_history.loc[len(load_history)] = ({
                'current_time': current_time,
                'original_load': original_load,
                'total_load': original_load + total_action_impact[current_time],
                'total_action_impact': total_action_impact[current_time]
            })
        else:
            load_history.loc[len(load_history)] = ({
                'current_time': current_time,
                'original_load': original_load,
                'total_load': original_load,
                'total_action_impact': 0
            })
        current_time += timedelta(hours=1)
        
    # Save load history
    load_history = load_history.sort_values(by='current_time')
    load_history_file = f'{result_dir}/building_loading_history.csv'
    load_history.to_csv(load_history_file, index=False)

    print(f'Test results and histories saved to {result_dir}')