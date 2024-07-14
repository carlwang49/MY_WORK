import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from logger_config import configured_logger as logger
from EVChargingEnv import EVChargingEnv
from collections import defaultdict
from utils import min_max_scaling
from ActionSpace import ActionSpace
from PriceEnvironment import PriceEnvironment

building_load_file = BUILDING_LOAD_FILE = '../Dataset/BuildingEnergyLoad/BuildingConsumptionLoad.csv'
Price_file_path = PRICE_FILE_PATH = '../Dataset/RTP/electricity_prices_from_201807010000_to_201812312359.csv'
alpha = ALPHA = 0.7
beta = BETA = 0.1
gamma = GAMMA = 0.1
max_load = MAX_LOAD = 1000  
min_load = MIN_LOAD = 0

class EVBuildingEnv(EVChargingEnv):
    def __init__(self, num_agents, start_time, end_time):
        super().__init__(num_agents, start_time, end_time)
        
        # set TOU price
        self.tou_price_in_weekday = [0.056] * 8 + [0.092] * 4 + [0.267] * 6 + [0.092] * 5 + [0.056] * 1
        self.tou_price_in_weekend = [0.056] * 24
        
        # init the maximum and minimum SoC
        self.SoC_upper_bound_dict = {f'agent_{i}': 0 for i in range(num_agents)}
        self.Soc_lower_bound_dict = {f'agent_{i}': 0 for i in range(num_agents)}

        # Initialize PriceEnvironment
        self.price_env = PriceEnvironment(Price_file_path, start_time, end_time)
        
        # set contract capacity
        self.contract_capacity = 800 
        
        # Initialize building load
        self.building_load = pd.read_csv(building_load_file, parse_dates=['Date'])
        self.building_load = self.set_building_time_range(start_time, end_time)
        self.count = None
        self.original_load = None
        
        # Initialize building load
        self.building_load = pd.read_csv(building_load_file, parse_dates=['Date'])
        self.building_load = self.set_building_time_range(start_time, end_time)
        self.load_dict = self.building_load.set_index('Date')['Total_Power(kWh)'].to_dict()
        self.date_list = [pd.Timestamp(date).to_pydatetime() for date in self.building_load['Date'].values]
        self.original_load = self.get_current_load(start_time)
        
        self.daily_stats = {}
        self.precompute_daily_stats()
 
        # Initialize the observation space and action space for each agent
        self.agents = [f'agent_{i}' for i in range(num_agents)]
        self.agents_status = {f'agent_{i}': False for i in range(num_agents)}
        self.current_parking = np.zeros(self.num_agents, dtype=bool)
        self.current_parking_number = 0  # Number of currently connected charging piles
        
        # Initialize the observation space and action space for each agent
        self.observation_spaces = np.zeros(4) 
        # self.action_values = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
        self.action_spaces = ActionSpace(-1, 1, (num_agents,))
        # self.action_spaces = {f'agent_{i}': DiscreteActionSpace(self.action_values) for i in range(num_agents)}  
        
        self.dones = {f'station_{i}': False for i in range(num_agents)}
        self.inactive_observation = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        
        # Calculate the average of the top 10% of historical peak electricity consumption
        sorted_load_history = self.building_load['Total_Power(kWh)'].sort_values(ascending=False)
        top_10_percent_index = int(len(sorted_load_history) * 0.25) # top 25% of historical peak electricity consumption
        top_10_percent_loads = sorted_load_history[:top_10_percent_index].copy()
        self.average_top_10_percent = np.mean(top_10_percent_loads) # Calculate the average of the top 10% of historical peak electricity consumption

        # Calculate the average of the top 10% of historical peak electricity consumption
        sorted_load_history = self.building_load['Total_Power(kWh)'].sort_values(ascending=False)
        top_10_percent_index = int(len(sorted_load_history) * 0.25) # top 25% of historical peak electricity consumption
        top_10_percent_loads = sorted_load_history[:top_10_percent_index].copy()
        self.average_top_10_percent = np.mean(top_10_percent_loads) # Calculate the average of the top 10% of historical peak electricity consumption

        # Calculate the maximum building load and the average building load
        self.max_load = self.building_load['Total_Power(kWh)'].max() 
        self.min_load = self.building_load['Total_Power(kWh)'].min()
        self.standard_deviation = self.building_load['Total_Power(kWh)'].std()
        self.average_load = self.building_load['Total_Power(kWh)'].mean() 
        self.min_load_diff = self.min_load - self.average_load
        self.max_load_diff = self.max_load - self.average_load
        
        # peak and valley load at current day
        self.curr_peak_load, self.curr_valley_load, self.curr_mean, self.curr_std = \
            self.get_current_peak_and_valley_load(self.start_time)

        # Store the building load history
        self.load_history = [] 
        
        # Initialize EV data for each charging pile
        self.ev_data = {
                f'agent_{i}': {
                'requestID': None,
                'arrival_time': None,
                'departure_time': None,
                'initial_soc': 0.0,
                'departure_soc': 0.0,
                'soc': 0.0,
                'time_before_soc_max': None,
                'time_before_soc_min': None,
            } for i in range(num_agents)
        }
    
    
    """set the building load time range for the environment"""
    def set_building_time_range(self, start_time: datetime, end_time: datetime):
        building_load = self.building_load[(self.building_load['Date'] >= start_time) & (self.building_load['Date'] <= end_time)].copy()
        building_load.sort_values(by='Date', inplace=True)
        return building_load
    
    def get_current_load(self, timestamp):
        return self.load_dict[timestamp]
    
    
    """set the building load time range for the environment"""
    def set_building_time_range(self, start_time: datetime, end_time: datetime):
        building_load = self.building_load[(self.building_load['Date'] >= start_time) & (self.building_load['Date'] <= end_time)].copy()
        building_load.sort_values(by='Date', inplace=True)
        return building_load
    
    """get daily statistics for the building load"""
    def precompute_daily_stats(self):
        self.building_load['date_only'] = self.building_load['Date'].dt.date
        self.building_load['time_only'] = self.building_load['Date'].dt.time

        for current_date in self.building_load['date_only'].unique():
            df_current_day = self.building_load[self.building_load['date_only'] == current_date].copy()
            df_current_day = df_current_day[(df_current_day['time_only'] >= pd.to_datetime('07:00:00').time()) & 
                                            (df_current_day['time_only'] <= pd.to_datetime('23:00:00').time())]
            max_load = df_current_day['Total_Power(kWh)'].max()
            min_load = df_current_day['Total_Power(kWh)'].min()
            mean_load = df_current_day['Total_Power(kWh)'].mean()
            std_load = df_current_day['Total_Power(kWh)'].std()
            self.daily_stats[current_date] = {
                'max': max_load,
                'min': min_load,
                'mean': mean_load,
                'std': std_load
            }
    
    """get observations for each agent in the environment"""
    @property
    def observation_space(self):
        return self.observation_spaces
    
    """get the action space for each agent in the environment"""
    @property
    def action_space(self):     
        return self.action_spaces
    

    """get state information for each agent in the environment"""
    def observe(self, current_time: datetime):

        # Get the normalized difference between the building load and the average of the top 10% of historical peak electricity consumption
        building_load = self.building_load[self.building_load['Date'] == current_time]['Total_Power(kWh)'].values[0]
        load_diff = building_load - self.curr_mean 
        normalized_load_diff = min_max_scaling(load_diff, self.min_load_diff, self.max_load_diff)
        
        # Get the current electricity price
        current_price = self.price_env.get_current_price(current_time)
            
        # Return the state information
        # return np.array([soc, normalized_load_diff, normalized_P_max_tk, normalized_P_min_tk, emergency, current_price], dtype=np.float32)
        return np.array([normalized_load_diff, current_price, self.timestamp.hour, self.timestamp.weekday()], dtype=np.float32)
    
    """reset the environment to the initial state"""
    def reset(self):
        
        # Reset the timestamp to the start time
        self.timestamp = self.start_time # Reset the timestamp to the start time
        self.count = None
        
        # peak and valley load at current day
        self.curr_peak_load, self.curr_valley_load, self.curr_mean, self.curr_std = self.get_current_peak_and_valley_load(self.start_time)
        
        # Initialize building load
        self.building_load = pd.read_csv(building_load_file, parse_dates=['Date'])
        self.building_load = self.set_building_time_range(self.start_time, self.end_time)
        self.load_dict = self.building_load.set_index('Date')['Total_Power(kWh)'].to_dict()
        self.date_list = [pd.Timestamp(date).to_pydatetime() for date in self.building_load['Date'].values]
        self.original_load = self.get_current_load(self.start_time)
        self.daily_stats = {}
        self.precompute_daily_stats()
 
        # Initialize the observation space and action space for each agent
        self.agents = [f'agent_{i}' for i in range(self.num_agents)]
        self.agents_status = {f'agent_{i}': False for i in range(self.num_agents)}
        self.observation_spaces = np.zeros(4)  
        
        # self.action_values = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
        self.action_spaces = ActionSpace(-1, 1, (self.num_agents,))
        # self.action_spaces = {f'agent_{i}': DiscreteActionSpace(self.action_values) for i in range(self.num_agents)}  
        
        self.dones = {f'station_{i}': False for i in range(self.num_agents)}
        
        self.current_parking = np.zeros(self.num_agents, dtype=bool)  # Whether the charging pile is connected
        self.current_parking_number = 0  # Number of currently connected charging piles
        self.SoC_upper_bound_list = [0 for _ in range(self.num_agents)]  # Upper bound of SoC
        self.SoC_lower_bound_list = [0 for _ in range(self.num_agents)]  # Lower bound of SoC
        
        self.load_history = [] 
        
        # Initialize EV data for each charging pile
        self.ev_data = {
                f'agent_{i}': {
                'requestID': None,
                'arrival_time': None,
                'departure_time': None,
                'initial_soc': 0.0,
                'departure_soc': 0.0,
                'soc': 0.0,
                'time_before_soc_max': None,
                'time_before_soc_min': None,
            } for i in range(self.num_agents)
        }
        
        # Reset the charging records and SoC history
        self.charging_records = pd.DataFrame(columns=['requestID', 
                                                      'arrival_time', 
                                                      'departure_time', 
                                                      'initial_soc', 
                                                      'departure_soc',
                                                      'final_soc', 
                                                      'charging_power', 
                                                      'charging_time']) 
        
        # Initialize a DataFrame to store SoC history
        self.soc_history = pd.DataFrame(columns=['requestID', 
                                                 'current_time', 
                                                 'soc', 
                                                 'SoC_upper_bound', 
                                                 'SoC_lower_bound'])
        
        observations = self.observe(self.start_time)
        
        return observations
    
    
    def calculate_reward(self, original_load, active_agent_ids, P_tk_dict: dict):

        rewards = {agent_id: 0 for agent_id in self.agents}
        current_price = self.price_env.get_current_price(self.timestamp)
        total_action_impact = sum(P_tk_dict.values())
        load_diff = original_load + total_action_impact - self.curr_mean
        penalty = original_load + total_action_impact - self.contract_capacity
        global_reward = -penalty if penalty > 0 else 0

        for agent_id in active_agent_ids:
            P_tk = P_tk_dict[agent_id]
            r_tk = -P_tk * current_price
            r_soc = abs(self.ev_data[agent_id]['soc'] - self.get_ev_reasonable_soc(agent_id, self.timestamp))
            rewards[agent_id] = (1-alpha) * r_tk - alpha * abs(P_tk + load_diff / len(self.agents)) + global_reward * gamma + beta * r_soc

        return rewards



    def step(self, actions, current_time: datetime, time_interval: int = 60):
        
        # Initialize the rewards, dones, infos, and observations
        rewards = defaultdict(int)
        dones = False
        infos = {}
        observations = {}
        P_tk_dict = {agent_id: 0 for agent_id in self.agents}
        
        # Get the original building load
        self.original_load = self.building_load[self.building_load['Date'] == current_time]['Total_Power(kWh)'].values[0].copy() 
        active_agent_ids = [agent_id for agent_id, status in self.agents_status.items() if status]
        
        for idx, agent_id in enumerate(self.agents):
            if self.agents_status[agent_id]:
                
                # Record the SoC history
                self.soc_history.loc[len(self.soc_history)] = ({
                    'requestID': self.ev_data[agent_id]['requestID'],  # Add 'requestID' to the soc_history DataFrame
                    'current_time': current_time, # Add the current time to the soc_history DataFrame
                    'soc': self.ev_data[agent_id]['soc'], # Add the current SoC to the soc_history DataFrame
                    'SoC_upper_bound': self.SoC_upper_bound_dict[agent_id], # Add the upper bound of SoC to the soc_history DataFrame
                    'SoC_lower_bound': self.Soc_lower_bound_dict[agent_id]  # Add the lower bound of SoC to the soc_history DataFrame
                })
                
                action = actions[idx]
                P_max_tk, P_min_tk, SoC_lower_bound, SoC_upper_bound = self.get_deb_constraints(agent_id, current_time + timedelta(hours=1))
                
                P_tk = (action + 1) / 2 * (P_max_tk - P_min_tk) + P_min_tk # Calculate the power output based on the action
                soc = (self.ev_data[agent_id]['soc'] * self.C_k + P_tk * (time_interval / 60)) / self.C_k 
                self.ev_data[agent_id]['soc'] = soc
                self.ev_data[agent_id]['soc'] = np.clip(soc, self.soc_min, self.soc_max)  # Ensure SoC is within a reasonable range
                self.Soc_lower_bound_dict[agent_id], self.SoC_upper_bound_dict[agent_id] = SoC_lower_bound, SoC_upper_bound
                
                # Record the power output for each charging pile
                P_tk_dict[agent_id] = P_tk 
        
        # Update the building load
        total_action_impact = sum(P_tk_dict.values())
        self.building_load.loc[self.building_load['Date'] == current_time, 'Total_Power(kWh)'] += total_action_impact
        
        self.load_history.append({
                'current_time': current_time,
                'original_load': self.original_load,
                'total_load': self.building_load.loc[self.building_load['Date'] == current_time, 'Total_Power(kWh)'].values[0],
                'total_action_impact': total_action_impact  
            })

        rewards = self.calculate_reward(self.original_load, active_agent_ids, P_tk_dict)

        observations = self.observe(current_time + timedelta(hours=1)) 
        
        self.timestamp = current_time + timedelta(minutes=time_interval) # Update the timestamp
        
        if self.timestamp.date() != current_time.date():
            # update the peak and valley load at the next day
            self.curr_peak_load, self.curr_valley_load, self.curr_mean, self.curr_std = self.get_current_peak_and_valley_load(self.timestamp)
        
        return observations, sum(rewards.values()), dones, infos


    """add an electric vehicle (EV) to the environment"""
    def add_ev(self, requestID, arrival_time, departure_time, initial_soc, departure_soc):
        avavailable_agents = [agent_id for agent_id, status in self.agents_status.items() if status == False]
        if len(avavailable_agents) > 0:
            selected_agent = np.random.choice(avavailable_agents)  # Randomly select an unconnected charging pile
            self.agents_status[selected_agent] = True  # Connect the selected charging pile
            self.Soc_lower_bound_dict[selected_agent] = self.SoC_upper_bound_dict[selected_agent] = initial_soc  # Set the lower bound of SoC
            
            # Initialize EV data for the selected charging pile
            self.ev_data[selected_agent] = {
                'requestID': requestID,
                'arrival_time': arrival_time,
                'departure_time': departure_time,
                'initial_soc': initial_soc,
                'departure_soc': departure_soc,
                'soc': initial_soc,
                'time_before_soc_max': arrival_time + timedelta(minutes=((self.soc_max - initial_soc) * self.C_k / self.max_charging_power) * 60), # Calculate the time before the SoC reaches the maximum value
                'time_before_soc_min': arrival_time + timedelta(minutes=((self.soc_min - initial_soc) * self.C_k / self.max_discharging_power) * 60), # Calculate the time before the SoC reaches the minimum value
            }
            
            # Record the SoC history
            self.soc_history.loc[len(self.soc_history)] = ({
                'requestID': self.ev_data[selected_agent]['requestID'],  # Add 'requestID' to the soc_history DataFrame
                'current_time': arrival_time, # Add the current time to the soc_history DataFrame
                'soc': self.ev_data[selected_agent]['soc'], # Add the current SoC to the soc_history DataFrame
                'SoC_upper_bound': self.SoC_upper_bound_dict[selected_agent], # Add the upper bound of SoC to the soc_history DataFrame
                'SoC_lower_bound': self.Soc_lower_bound_dict[selected_agent]  # Add the lower bound of SoC to the soc_history DataFrame
            })
            
            # Log the EV data
            # logger.bind(console=True).info(f"requestID: {self.ev_data[selected_agent]['requestID']}")
            # logger.bind(console=True).info(f"initial_soc: {self.ev_data[selected_agent]['initial_soc']}")
            # logger.bind(console=True).info(f"departure_soc: {self.ev_data[selected_agent]['departure_soc']}")
            # logger.bind(console=True).info(f"arrival_time: {self.ev_data[selected_agent]['arrival_time']}")
            # logger.bind(console=True).info(f"departure_time: {self.ev_data[selected_agent]['departure_time']}")
            # logger.bind(console=True).info(f"time_before_soc_max: {self.ev_data[selected_agent]['time_before_soc_max']}")
            # logger.bind(console=True).info(f"time_before_soc_min: {self.ev_data[selected_agent]['time_before_soc_min']}")
            # logger.bind(console=True).info('-' * 50)
        else:
            logger.bind(console=True).warning("No available charging piles.")
            return None
        
    def remove_ev(self, agent_id):
        if self.agents_status[agent_id]:
            self.agents_status[agent_id] = False  # Disconnect the selected charging pile
            
            # Record the charging history
            self.charging_records.loc[len(self.charging_records)] = {
                'requestID': self.ev_data[agent_id]['requestID'],
                'arrival_time': self.ev_data[agent_id]['arrival_time'],
                'departure_time': self.ev_data[agent_id]['departure_time'],
                'initial_soc': self.ev_data[agent_id]['initial_soc'],
                'departure_soc': self.ev_data[agent_id]['departure_soc'], # Add 'departure_soc' to the charging_records DataFrame
                'final_soc': self.ev_data[agent_id]['soc'],
                'charging_power': (self.ev_data[agent_id]['soc'] - self.ev_data[agent_id]['initial_soc']) * self.C_k, # kWh
                'charging_time': (self.ev_data[agent_id]['departure_time'] - self.ev_data[agent_id]['arrival_time']).seconds / 3600 # Convert seconds to hours
            }
            
            # Record the SoC history
            self.soc_history.loc[len(self.soc_history)] = ({
                'requestID': self.ev_data[agent_id]['requestID'],  # Add 'requestID' to the soc_history DataFrame
                'current_time': self.ev_data[agent_id]['departure_time'], # Add the current time to the soc_history DataFrame
                'soc': self.ev_data[agent_id]['soc'], # Add the current SoC to the soc_history DataFrame
                'SoC_upper_bound': self.SoC_upper_bound_dict[agent_id], # Add the upper bound of SoC to the soc_history DataFrame
                'SoC_lower_bound': self.Soc_lower_bound_dict[agent_id]  # Add the lower bound of SoC to the soc_history DataFrame
            })
            
            
            # Reset EV data for the selected charging pile
            self.ev_data[agent_id] = {'requestID': None, 
                                    'arrival_time': None, 
                                    'departure_time': None, 
                                    'initial_soc': 0.0, 
                                    'soc': 0.0, 
                                    'time_before_soc_max': None,
                                    'time_before_soc_min': None,} 
        else:
            logger.bind(console=True).warning(f"Charging pile {agent_id} is not connected.")
            return None
            

    def get_ev_reasonable_soc(self, agent_id, current_time: datetime):
        """Get the reasonable SoC based on the current time"""
        
        reasonable_soc = (current_time - self.ev_data[agent_id]['arrival_time']).seconds \
            / (self.ev_data[agent_id]['departure_time'] - self.ev_data[agent_id]['arrival_time']).seconds \
                * (self.ev_data[agent_id]['departure_soc'] - self.ev_data[agent_id]['initial_soc']) + self.ev_data[agent_id]['initial_soc']
        
        return round(reasonable_soc, 2)       
    

    def get_current_peak_and_valley_load(self, timestamp):
        current_date = timestamp.date()
        if current_date in self.daily_stats:
            return self.daily_stats[current_date]['max'], self.daily_stats[current_date]['min'], self.daily_stats[current_date]['mean'], self.daily_stats[current_date]['std']
        else:
            return None, None, None, None