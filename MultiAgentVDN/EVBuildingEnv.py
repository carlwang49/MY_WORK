import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from logger_config import configured_logger as logger
from EVChargingEnv import EVChargingEnv
from collections import defaultdict
from gym.spaces import Box, Discrete
from utilities import min_max_scaling, standardize

building_load_file = BUILDING_LOAD_FILE = '../Dataset/BuildingEnergyLoad/BuildingConsumptionLoad.csv'
alpha = ALPHA = 0.5

class ActionSpace:
    """Define the action space for each agent in the environment"""
    def __init__(self, low, high, shape):
        self.low = low
        self.high = high
        self.shape = shape

    def sample(self):
        sample = np.random.uniform(self.low, self.high, self.shape)
        return round(float(sample), 4)
    
    def size(self):
        return self.shape[0]
    

class EVBuildingEnv(EVChargingEnv):
    def __init__(self, num_agents, start_time, end_time):
        super().__init__(num_agents, start_time, end_time)
        
        # set TOU price
        self.tou_price_in_weekday = [0.056] * 8 + [0.092] * 4 + [0.267] * 6 + [0.092] * 5 + [0.056] * 1
        self.tou_price_in_weekend = [0.056] * 24
        self.SoC_upper_bound_dict = {f'agent_{i}': 0 for i in range(num_agents)}
        self.Soc_lower_bound_dict = {f'agent_{i}': 0 for i in range(num_agents)}

        # Initialize building load
        self.building_load = pd.read_csv(building_load_file, parse_dates=['Date'])
        self.building_load = self.set_building_time_range(start_time, end_time)
        
        self.count = None
        self.original_load = None
 
        # Initialize the observation space and action space for each agent
        self.agents = [f'agent_{i}' for i in range(num_agents)]
        self.agents_status = {f'agent_{i}': False for i in range(num_agents)}
        self.observation_spaces = [Box(low=0, high=1, shape=(5,), dtype=np.float32) for i in range(num_agents)]
        self.action_values = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
        self.action_spaces = [Discrete(len(self.action_values)) for i in range(num_agents)]
        self.dones = {f'station_{i}': False for i in range(num_agents)}
        
        # Calculate the average of the top 10% of historical peak electricity consumption
        sorted_load_history = self.building_load['Total_Power(kWh)'].sort_values(ascending=False)
        top_10_percent_index = int(len(sorted_load_history) * 0.1)
        top_10_percent_loads = sorted_load_history[:top_10_percent_index].copy()
        self.average_top_10_percent = np.mean(top_10_percent_loads) # Calculate the average of the top 10% of historical peak electricity consumption
        
        # Calculate the maximum building load and the average building load
        self.max_load = self.building_load['Total_Power(kWh)'].max() 
        self.min_load = self.building_load['Total_Power(kWh)'].min()
        self.standard_deviation = self.building_load['Total_Power(kWh)'].std()
        self.average_load = self.building_load['Total_Power(kWh)'].mean() 
        self.min_load_diff = self.min_load - self.average_top_10_percent
        self.max_load_diff = self.max_load - self.average_top_10_percent
        
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
                'connected': False
            } for i in range(num_agents)
        }
    
    """set the building load time range for the environment"""
    def set_building_time_range(self, start_time: datetime, end_time: datetime):
        building_load = self.building_load[(self.building_load['Date'] >= start_time) & (self.building_load['Date'] <= end_time)].copy()
        building_load.sort_values(by='Date', inplace=True)
        return building_load
    
    
    """get observations for each agent in the environment"""
    def observation_space(self, agent_id):
        return self.observation_spaces[agent_id]

    """get the action space for each agent in the environment"""
    def action_space(self, agent_id):     
        return self.action_spaces[agent_id]
    

    """get state information for each agent in the environment"""
    def observe(self, agent_id, current_time: datetime):
        # If the current time is the start time, return the initial state
        if current_time == self.start_time or self.agents_status[f'agent_{agent_id}'] == False:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
    
        # Get the current SoC
        soc = round(self.ev_data[f'agent_{agent_id}']['soc'], 2)
        
        # Get the normalized difference between the building load and the average of the top 10% of historical peak electricity consumption
        building_load = self.building_load[self.building_load['Date'] == current_time]['Total_Power(kWh)'].values[0]
        load_diff = building_load - self.average_top_10_percent
        normalized_load_diff = min_max_scaling(load_diff, self.min_load_diff, self.max_load_diff)
        
        # Get the maximum and minimum SoC based on the current time
        P_max_tk, P_min_tk, _, _ = self.get_deb_constraints(f'agent_{agent_id}', current_time + timedelta(hours=1))
        normalized_P_max_tk = min_max_scaling(P_max_tk, self.max_discharging_power, self.max_charging_power)
        normalized_P_min_tk = min_max_scaling(P_min_tk, self.max_discharging_power, self.max_charging_power)
        
        # Get the time interval before EV's leaving time
        emergency = self.ev_data[f'agent_{agent_id}']['departure_time'].hour - current_time.hour
        emergency = float(emergency)
        
        # Return the state information
        return [soc, normalized_load_diff, normalized_P_max_tk, normalized_P_min_tk, emergency]
    
    
    """reset the environment to the initial state"""
    def reset(self):
        
        # Reset the timestamp to the start time
        self.timestamp = self.start_time # Reset the timestamp to the start time
        self.count = None
        
        # Initialize the environment
        self.building_load = pd.read_csv(building_load_file, parse_dates=['Date'])
        self.building_load = self.set_building_time_range(self.start_time, self.end_time)
        self.original_load = None
 
        # Initialize the observation space and action space for each agent
        self.agents = [f'agent_{i}' for i in range(self.num_agents)]
        self.agents_status = {f'agent_{i}': False for i in range(self.num_agents)}
        self.observation_spaces = {f'agent_{i}': np.zeros(5) for i in range(self.num_agents)} # SoC, building load, P_max_tk, P_min_tk
        self.action_spaces = {f'agent_{i}': ActionSpace(-1, 1, (1,)) for i in range(self.num_agents)}  
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
                'connected': False
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
        
        observations = [
            self.observe(agent, self.start_time)
            for agent in range(len(self.agents))
        ]
        return observations
    
    
    def calculate_reward(self, original_load, P_tk_dict: dict):

        rewards = [0 for _ in range(self.num_agents)]
        current_price = self.tou_price_in_weekday[self.timestamp.hour] if self.timestamp.weekday() < 5 else self.tou_price_in_weekend[self.timestamp.hour]
        
        for idx ,agent_id in enumerate(self.agents):
            if self.agents_status[agent_id]:
                P_tk = P_tk_dict[agent_id]
                r_tk = -P_tk * current_price
                rewards[idx] = r_tk
                
        return rewards



    def step(self, actions, current_time: datetime, time_interval: int = 60):
        
        # Convert the actions to power outputs
        actions = [self.action_values[int(a)] for a in actions]
        
        # Initialize the rewards, dones, infos, and observations
        rewards = []
        dones = [False for _ in self.agents]
        infos = []
        observations = []
        P_tk_dict = {agent_id: 0 for agent_id in self.agents}
        
        # Get the original building load
        self.original_load = self.building_load[self.building_load['Date'] == current_time]['Total_Power(kWh)'].values[0].copy() 
        
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

        rewards = self.calculate_reward(self.original_load, P_tk_dict)

        observations = [
            self.observe(agent, self.timestamp + timedelta(hours=1))
            for agent in range(len(self.agents))
        ]
        
        self.timestamp = current_time + timedelta(minutes=time_interval) # Update the timestamp
        
        return observations, rewards, dones, infos


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
                'connected': True,
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
                                    'time_before_soc_min': None,
                                    'connected': False} 
        else:
            logger.bind(console=True).warning(f"Charging pile {agent_id} is not connected.")
            return None
            
