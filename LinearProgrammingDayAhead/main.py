import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
from scipy.optimize import linprog
from logger_config import configured_logger as logger
from utils import create_result_dir, get_num_of_evs_in_each_hours, get_tou_price, get_rtp_price

# 設定 contract_capacity 和 capacity_price
contract_capacity = 500
capacity_price = 15

# Define the start and end date of the EV request data
start_date = START_DATE = '2018-07-01'
end_date = END_DATE = '2018-10-01'

# Define the start and end time of the EV request data
start_time = START_TIME = datetime(2018, 7, 1)
end_time = END_TIME = datetime(2018, 10, 1)

num_agents = NUM_AGENTS = 10
parking_data_path = PARKING_DATA_PATH = f'../Dataset/Sim_Parking/ev_parking_data_from_2018-07-01_to_2018-12-31_{NUM_AGENTS}.csv'
building_load_file = BUILDING_LOAD_FILE = '../Dataset/BuildingEnergyLoad/BuildingConsumptionLoad.csv'

min_soc = MIN_SOC = 0.2
max_soc = MAX_SOC = 0.9
c_k = C_K = 75
eta = ETA = 0.85
max_charging_power = MAX_CHARGING_POWER = 120
max_discharging_power = MAX_DISCHARGING_POWER = -120

"""set the building load time range for the environment"""
def set_building_time_range(building_load, start_time: datetime, end_time: datetime):
    building_load = building_load[(building_load['Date'] >= start_time) & (building_load['Date'] <= end_time)].copy()
    building_load.sort_values(by='Date', inplace=True)
    return building_load

if __name__ == '__main__':
    """
    Run a simulation of the EV charging environment with random EV request data and random charging power selection.
    """
    # Initialize a DataFrame to store charging records and SoC history
    charging_records = pd.DataFrame(columns=['requestID', 
                                            'arrival_time', 
                                            'departure_time', 
                                            'initial_soc', 
                                            'departure_soc',
                                            'final_soc', 
                                            'charging_power', 
                                            'charging_time']) 
    
    # Initialize a DataFrame to store SoC history
    soc_history = pd.DataFrame(columns=['requestID', 
                                        'current_time', 
                                        'soc', 
                                        'current_energy',
                                        'charging_power/discharging_power',
                                        'cost',
                                        'total_cost']) 
    
    # Prepare EV request data
    day_ahead_schedule_df = pd.read_csv('../Dataset/Sim_Parking/ev_parking_data_from_2018-07-01_to_2018-12-31.csv')
    day_ahead_schedule_df['date'] = pd.to_datetime(day_ahead_schedule_df['date']).dt.date
    day_ahead_schedule_df['arrival_time'] = pd.to_datetime(day_ahead_schedule_df['arrival_time'])
    day_ahead_schedule_df['departure_time'] = pd.to_datetime(day_ahead_schedule_df['departure_time'])
    
    # Initialize building load
    building_load = pd.read_csv(building_load_file, parse_dates=['Date'])
    building_load = set_building_time_range(building_load, start_time, end_time)
    building_load['Date'] = pd.to_datetime(building_load['Date'])
    original_load = None
    load_history = pd.DataFrame(columns=['current_time', 'original_load', 'total_load', 'total_action_impact'])
    total_action_impact = defaultdict(float)
    
    # Create a directory for storing results of the simulation
    result_dir = create_result_dir(f'DayAheadSchedule_agent{NUM_AGENTS}')
    
    real_time_price = get_rtp_price(start_time, end_time)
    
    current_time = start_time  
    while current_time < end_time:
        
        number_of_ev_in_hours = get_num_of_evs_in_each_hours(day_ahead_schedule_df, current_time)
        
        # get EV request list in current date
        ev_request_list = day_ahead_schedule_df[day_ahead_schedule_df['date'] == current_time.date()].to_dict(orient='records') 
        for ev_request in ev_request_list:
            
            # Get information about the EV request
            initial_soc, final_soc = ev_request['initial_soc'], ev_request['departure_soc']
            arrival_time, departure_time = ev_request['arrival_time'].hour, ev_request['departure_time'].hour
            num_intervals = departure_time - arrival_time
            print(num_intervals)
            # Calculate the maximum and minimum charging power
            max_powers = [max_charging_power / number_of_ev_in_hours[hour] for hour in range(arrival_time, departure_time)]
            min_powers = [max_discharging_power / number_of_ev_in_hours[hour] for hour in range(arrival_time, departure_time)]
            
            # Calculate the energy requirements
            initial_energy = initial_soc * c_k
            required_energy = final_soc * c_k
            min_energy = min_soc * c_k
            max_energy = max_soc * c_k

            # Calculate the time intervals
            time_intervals = list(range(arrival_time, departure_time))
            # tou_price = get_tou_price(current_time)
            # relevant_prices = tou_price[arrival_time:departure_time]
            tou_price = real_time_price[(real_time_price['datetime'] >= ev_request['arrival_time']) & (real_time_price['datetime'] < ev_request['departure_time'])]
            relevant_prices = tou_price['average_price'].values
            
            # Define the linear programming problem
            c = np.concatenate((relevant_prices, -np.array(relevant_prices)))
            
            # Ensure that the total energy change is equal to the required energy change
            A_eq = np.concatenate((np.ones(num_intervals), -np.ones(num_intervals))).reshape(1, -1)
            b_eq = np.array([required_energy - initial_energy])
            
            # Charge and discharge power constraints
            A_ub = np.zeros((2 * num_intervals + 2 * num_intervals, 2 * num_intervals))
            b_ub = np.zeros(2 * num_intervals + 2 * num_intervals)
            
            for i in range(num_intervals):
                # charging power <= max_powers[i]
                A_ub[i, i] = 1 
                A_ub[num_intervals + i, num_intervals + i] = 1  
                
                # discharging power <= -min_powers[i]
                # (note that the discharging power is negative, so we need to take the negative)
                b_ub[i] = max_powers[i]
                b_ub[num_intervals + i] = -min_powers[i]
            
            # SoC constraints: ensure that SoC is not less than min_soc and not greater than max_soc 
            for i in range(num_intervals):
                A_ub[2 * num_intervals + i, :i + 1] = 1
                A_ub[2 * num_intervals + i, num_intervals:num_intervals + i + 1] = -1
                b_ub[2 * num_intervals + i] = max_energy - initial_energy
                
                A_ub[3 * num_intervals + i, :i + 1] = -1
                A_ub[3 * num_intervals + i, num_intervals:num_intervals + i + 1] = 1
                b_ub[3 * num_intervals + i] = initial_energy - min_energy

            # Variable upper and lower limits
            bounds = [(0, max_powers[i]) for i in range(num_intervals)] + [(0, -min_powers[i]) for i in range(num_intervals)]
            
            # Solve the linear programming problem
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method='highs')
            
            if res.success:
                current_energy = initial_energy
                total_cost = 0
                current_soc = initial_soc
                prev_soc = 0
                prev_energy = 0
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
                        
                    prev_energy = current_energy # store the previous energy
                    prev_soc = current_soc # store the previous SoC
                    
                    current_energy += energy if action == 'charge' else -energy # update the current energy
                    current_soc = current_energy / c_k # update the current SoC
                    total_cost += cost # update the total cost
                    
                    soc_history.loc[len(soc_history)] = ({
                        'requestID': ev_request['requestID'],    
                        'current_time': current_time + timedelta(hours=time_intervals[i]),
                        'soc': round(prev_soc, 2),    
                        'current_energy': round(prev_energy, 2),
                        'charging_power/discharging_power': -energy if action == 'discharge' else energy,
                        'cost': round(cost, 2),
                        'total_cost': round(total_cost, 2)
                    })
                    total_action_impact[current_time + timedelta(hours=time_intervals[i])] += energy if action == 'charge' else -energy
                    soc_history.loc[len(soc_history)] = ({
                        'requestID': ev_request['requestID'],    
                        'current_time': current_time + timedelta(hours=time_intervals[-1]+1),
                        'soc': round(current_soc, 2),    
                        'current_energy': round(current_energy, 2),
                        'charging_power/discharging_power': 0,
                        'cost': 0,
                        'total_cost': round(total_cost, 2)
                    })
                    
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

    soc_history.to_csv(f'../Result/{result_dir}/soc_history.csv', index=False)
    charging_records.to_csv(f'../Result/{result_dir}/charging_records.csv', index=False)
    
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
        
    # save load history
    # Sort the DataFrame by current_time
    load_history = load_history.sort_values(by='current_time')
    load_history_file = f'{result_dir}/building_loading_history.csv'
    load_history.to_csv(load_history_file, index=False)