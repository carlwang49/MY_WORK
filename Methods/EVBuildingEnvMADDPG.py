import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from logger_config import configured_logger as logger
from EVChargingEnv import EVChargingEnv
from collections import defaultdict

building_load_file = BUILDING_LOAD_FILE = '../Dataset/BuildingEnergyLoad/BuildingConsumptionLoad.csv'

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

        # Initialize building load
        self.start_time = start_time
        self.end_time = end_time
        self.timestamp = start_time
        self.building_load = pd.read_csv(building_load_file, parse_dates=['Date'])
        self.building_load = self.set_building_time_range(start_time, end_time)
        
        self.count = None
        self.original_load = None
 
        # Initialize the observation space and action space for each agent
        self.agents = [f'agent_{i}' for i in range(num_agents)]
        self.agents_status = {f'agent_{i}': False for i in range(num_agents)}
        self.observation_spaces = {f'agent_{i}': np.zeros(4) for i in range(num_agents)} # SoC, building load, P_max_tk, P_min_tk
        self.action_spaces = {f'agent_{i}': ActionSpace(-1, 1, (1,)) for i in range(num_agents)}  
        self.dones = {f'station_{i}': False for i in range(num_agents)}
        
        # Calculate the average of the top 10% of historical peak electricity consumption
        sorted_load_history = self.building_load['Total_Power(kWh)'].sort_values(ascending=False)
        top_10_percent_index = int(len(sorted_load_history) * 0.1)
        top_10_percent_loads = sorted_load_history[:top_10_percent_index].copy()
        
        self.average_top_10_percent = np.mean(top_10_percent_loads) # Calculate the average of the top 10% of historical peak electricity consumption
        self.max_load = self.building_load['Total_Power(kWh)'].max() # Calculate the maximum building load
        self.average_load = self.building_load['Total_Power(kWh)'].mean() # Calculate the average building load
        
        self.load_history = [] # Store the building load history
        
        self.ev_data = {
            f'agent_{i}': {
            'requestID': None,
            'arrival_time': None,
            'departure_time': None,
            'initial_soc': 0.0,
            'soc': 0.0,
            'time_before_soc_max': None,
            'time_before_soc_min': None,
            }
            for i in range(num_agents)
        }
    
    def set_building_time_range(self, start_time: datetime, end_time: datetime):
        building_load = self.building_load[(self.building_load['Date'] >= start_time) & (self.building_load['Date'] <= end_time)].copy()
        building_load.sort_values(by='Date', inplace=True)
        return building_load
    
    
    def observation_space(self, agent_id):
        return self.observation_spaces[agent_id]


    def action_space(self, agent_id):     
        return self.action_spaces[agent_id]
    

    def observe(self, agent_id, current_time: datetime):
        # If the current time is the start time, return the initial state
        if current_time == self.start_time:
            return np.array([0, 0, 0, 0], dtype=np.float32
                            )
        if self.agents_status[agent_id] == False:
            return np.array([0, 0, 0, 0], dtype=np.float32)
        
        # Get the current SoC
        soc = self.ev_data[agent_id]['soc'] 
        # Get the building load at the current time
        building_load = self.building_load[self.building_load['Date'] == current_time]['Total_Power(kWh)'].values[0] 
        # Get the maximum and minimum SoC based on the current time
        P_max_tk, P_min_tk, _, _ = self.get_deb_constraints(agent_id, current_time)
        
        return np.array([soc, building_load / self.max_load, P_max_tk / self.max_charging_power, P_min_tk / self.max_charging_power], dtype=np.float32)
    
    def reset(self):
        # Initialize the environment
        self.load_history = [] 
        self.agents_status = {f'agent_{i}': False for i in range(self.num_agents)}
        self.building_load = pd.read_csv('../Dataset/BuildingEnergyLoad/BuildingConsumptionLoad.csv', parse_dates=['Date'])
        self.building_load = self.set_building_time_range(self.start_time, self.end_time)
        self.original_load = None
        self.dones = {f'station_{i}': False for i in range(self.num_agents)}
        self.current_parking = np.zeros(self.num_agents, dtype=bool)
        self.current_parking_number = 0
        
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
        
        # Initialize the EV data for each agent 
        self.ev_data = {
            f'agent_{i}': {
            'requestID': None,
            'arrival_time': None,
            'departure_time': None,
            'initial_soc': 0.0,
            'soc': 0.0,
            'time_before_soc_max': None,
            'time_before_soc_min': None,
            }
            for i in range(self.num_agents)
        }
        
        observations = {
            agent: self.observe(agent, self.start_time)
            for agent in self.agents
        }
        return observations
    
    
    def calculate_reward(self, original_load, P_tk_dict: dict):
        contract_capacity = self.average_load
        total_load = original_load + sum(P_tk_dict.values())
        total_discharge = sum(P_tk for P_tk in P_tk_dict.values() if P_tk < 0)

        rewards = {agent_id: 0 for agent_id in self.agents}

        for agent_id in self.agents:
            if self.agents_status[agent_id]:
                P_tk = P_tk_dict[agent_id]
                
                # 基本懲罰：如果總負載超過合約容量
                if total_load > contract_capacity:
                    penalty = -10 * (total_load - contract_capacity)
                    rewards[agent_id] += penalty
                
                # 放電獎勵：根據放電行為，並且總負載接近合約容量
                if P_tk < 0:  # Discharging
                    if total_discharge != 0:
                        discharge_reward = (abs(P_tk) / abs(total_discharge)) * 10
                        rewards[agent_id] += discharge_reward
                
                # 集體行為獎勵：所有智能體的總負載接近合約容量
                proximity_reward = 10 / (1 + np.exp(-0.1 * (total_load - contract_capacity)))
                rewards[agent_id] += proximity_reward

        return rewards



    def step(self, actions, current_time: datetime, time_interval: int = 60):
        
        # Initialize the rewards, dones, infos, and observations
        rewards = defaultdict(int)
        dones = {agent_id: False for agent_id in self.agents}
        infos = {}
        observations = {}
        P_tk_dict = {agent_id: 0 for agent_id in self.agents}
        
        # Get the original building load
        last_time = current_time - timedelta(minutes=time_interval) if current_time != self.start_time else current_time
        self.original_load = self.building_load[self.building_load['Date'] == last_time]['Total_Power(kWh)'].values[0].copy() 
        
        # logger.debug(f"actions: {actions}")
        for agent_id in self.agents:
            if self.agents_status[agent_id]:
                action = actions[agent_id]
                P_max_tk, P_min_tk, SoC_lower_bound, SoC_upper_bound = self.get_deb_constraints(agent_id, current_time)
                P_tk = (action + 1) / 2 * (P_max_tk - P_min_tk) + P_min_tk # Calculate the power output based on the action
                soc = (self.ev_data[agent_id]['soc'] * self.C_k + P_tk * (time_interval / 60)) / self.C_k 
                self.ev_data[agent_id]['soc'] = soc
                self.ev_data[agent_id]['soc'] = np.clip(soc, self.soc_min, self.soc_max)  # Ensure SoC is within a reasonable range

                P_tk_dict[agent_id] = P_tk
                # Record the SoC history
                self.soc_history.loc[len(self.soc_history)] = ({
                    'requestID': self.ev_data[agent_id]['requestID'],  # Add 'requestID' to the soc_history DataFrame
                    'current_time': current_time, # Add the current time to the soc_history DataFrame
                    'soc': soc, # Add the current SoC to the soc_history DataFrame
                    'SoC_upper_bound': SoC_upper_bound, # Add the upper bound of SoC to the soc_history DataFrame
                    'SoC_lower_bound': SoC_lower_bound  # Add the lower bound of SoC to the soc_history DataFrame
                })
                
            if current_time == self.ev_data[agent_id]['departure_time']:
                self.agents_status[agent_id] = False
                dones[agent_id] = True
                self.remove_ev(agent_id)
                self.current_parking_number -= 1
                self.count -= 1
            
        self.previous_parking_number = self.current_parking_number - self.count
        total_action_impact = sum(P_tk_dict.values())
        self.building_load.loc[self.building_load['Date'] == current_time, 'Total_Power(kWh)'] += total_action_impact
        
        self.load_history.append({
                'current_time': current_time,
                'original_load': self.original_load,
                'total_load': self.original_load + total_action_impact,
                'total_action_impact': total_action_impact  
            })

        rewards = self.calculate_reward(self.original_load, P_tk_dict)
        observations = {
            agent: self.observe(agent, current_time)
            for agent in self.agents
        }
        self.timestamp = current_time + timedelta(minutes=time_interval) # Update the timestamp
        return observations, rewards, dones, infos

    def add_ev(self, requestID, arrival_time, departure_time, initial_soc, departure_soc):
        avavailable_agents = [agent_id for agent_id, status in self.agents_status.items() if status == False]
        if len(avavailable_agents) > 0:
            selected_agent = np.random.choice(avavailable_agents)  # Randomly select an unconnected charging pile
            self.agents_status[selected_agent] = True  # Connect the selected charging pile
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
            # # Log the EV data
            # logger.info(f"requestID: {self.ev_data[selected_agent]['requestID']}")
            # logger.info(f"initial_soc: {self.ev_data[selected_agent]['initial_soc']}")
            # logger.info(f"departure_soc: {self.ev_data[selected_agent]['departure_soc']}")
            # logger.info(f"arrival_time: {self.ev_data[selected_agent]['arrival_time']}")
            # logger.info(f"departure_time: {self.ev_data[selected_agent]['departure_time']}")
            # logger.info(f"time_before_soc_max: {self.ev_data[selected_agent]['time_before_soc_max']}")
            # logger.info(f"time_before_soc_min: {self.ev_data[selected_agent]['time_before_soc_min']}")
            # logger.info('-' * 50)
        else:
            logger.warning("No available charging piles.")
            return None
        
    def remove_ev(self, agent_id):
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
        
        # Reset EV data for the selected charging pile
        self.ev_data[agent_id] = {'requestID': None, 
                                'arrival_time': None, 
                                'departure_time': None, 
                                'initial_soc': 0.0, 
                                'soc': 0.0, 
                                'time_before_soc_max': None,
                                'time_before_soc_min': None,
                                'connected': False} 
        
