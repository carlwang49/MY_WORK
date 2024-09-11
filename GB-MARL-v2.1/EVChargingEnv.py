import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from logger_config import configured_logger as logger
from dotenv import load_dotenv
import os
load_dotenv()

class EVChargingEnv:
    def __init__(self, num_agents, start_time, end_time):  
        self.num_agents = num_agents
        self.current_parking = np.zeros(num_agents, dtype=bool)  # Whether the charging pile is connected
        self.current_parking_number = 0  # Number of currently connected charging piles
        self.SoC_upper_bound_list = [0 for _ in range(num_agents)]  # Upper bound of SoC
        self.SoC_lower_bound_list = [0 for _ in range(num_agents)]  # Lower bound of SoC
        
        # Initialize the time range
        self.start_time = start_time
        self.end_time = end_time
        self.timestamp = start_time

        # Define charging pile power constraints
        self.max_charging_power = int(os.getenv('MAX_CHARGING_POWER', 150))  
        self.max_discharging_power = int(os.getenv('MAX_DISCHARGING_POWER', -150))  
        self.C_k = float(os.getenv('BATTERY_CAPACITY', 60))  
        self.eta = float(os.getenv('CHARGING_EFFICIENCY', 0.95))  
        self.soc_max = float(os.getenv('SOC_MAX', 0.8))  
        self.soc_min = float(os.getenv('SOC_MIN', 0.2))  

        # Initialize EV data for each charging pile
        self.ev_data = [{'requestID': None, 
                         'arrival_time': None, 
                         'departure_time': None, 
                         'initial_soc': 0.0, 
                         'departure_soc': 0.0,
                         'soc': 0.0, 
                         'time_before_soc_max': None,
                         'time_before_soc_min': None,
                         'connected': False} 
                        for _ in range(self.num_agents)]
        
        # Initialize a DataFrame to store charging records and SoC history
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
        
        
    def add_ev(self, requestID, arrival_time, departure_time, initial_soc, departure_soc):
        available_piles = np.where(self.current_parking == False)[0]  # Find indices of unconnected charging piles
        if len(available_piles) > 0:
            selected_pile = np.random.choice(available_piles)  # Randomly select an unconnected charging pile
            self.current_parking[selected_pile] = True  # Connect the selected charging pile
            self.SoC_lower_bound_list[selected_pile] = self.SoC_upper_bound_list[selected_pile] = initial_soc  # Set the lower bound of SoC
            
            # Initialize EV data for the selected charging pile
            self.ev_data[selected_pile] = {
                'requestID': requestID,
                'arrival_time': arrival_time,
                'departure_time': departure_time,
                'initial_soc': initial_soc,
                'departure_soc': departure_soc,
                'soc': initial_soc,
                'time_before_soc_max': arrival_time + timedelta(minutes=((self.soc_max - initial_soc) * self.C_k / self.max_charging_power) * 60), # Calculate the time before the SoC reaches the maximum value
                'time_before_soc_min': arrival_time + timedelta(minutes=((self.soc_min - initial_soc) * self.C_k / self.max_discharging_power) * 60), # Calculate the time before the SoC reaches the minimum value
                'connected': True # Indicate that the charging pile is connected
            }
            
            # Record the SoC history
            self.soc_history.loc[len(self.soc_history)] = ({
                'requestID': self.ev_data[selected_pile]['requestID'],  # Add 'requestID' to the soc_history DataFrame
                'current_time': arrival_time, # Add the current time to the soc_history DataFrame
                'soc': self.ev_data[selected_pile]['soc'], # Add the current SoC to the soc_history DataFrame
                'SoC_upper_bound': self.SoC_upper_bound_list[selected_pile], # Add the upper bound of SoC to the soc_history DataFrame
                'SoC_lower_bound': self.SoC_lower_bound_list[selected_pile]  # Add the lower bound of SoC to the soc_history DataFrame
            })
            
            # Log the EV data
            # logger.bind(console=True).info(f"requestID: {self.ev_data[selected_pile]['requestID']}")
            # logger.bind(console=True).info(f"initial_soc: {self.ev_data[selected_pile]['initial_soc']}")
            # logger.bind(console=True).info(f"departure_soc: {self.ev_data[selected_pile]['departure_soc']}")
            # logger.bind(console=True).info(f"arrival_time: {self.ev_data[selected_pile]['arrival_time']}")
            # logger.bind(console=True).info(f"departure_time: {self.ev_data[selected_pile]['departure_time']}")
            # logger.bind(console=True).info(f"time_before_soc_max: {self.ev_data[selected_pile]['time_before_soc_max']}")
            # logger.bind(console=True).info(f"time_before_soc_min: {self.ev_data[selected_pile]['time_before_soc_min']}")
            # logger.bind(console=True).info('-' * 50)
        else:
            logger.bind(console=True).warning("No available charging piles.")
            return None
        
    def remove_ev(self, agent_idx):
        if self.current_parking[agent_idx]:
            self.current_parking[agent_idx] = False  # Disconnect the selected charging pile
            # Record the charging history
            self.charging_records.loc[len(self.charging_records)] = {
                'requestID': self.ev_data[agent_idx]['requestID'],
                'arrival_time': self.ev_data[agent_idx]['arrival_time'],
                'departure_time': self.ev_data[agent_idx]['departure_time'],
                'departure_soc': self.ev_data[agent_idx]['departure_soc'], # Add 'departure_soc' to the charging_records DataFrame
                'initial_soc': self.ev_data[agent_idx]['initial_soc'],
                'final_soc': self.ev_data[agent_idx]['soc'],
                'charging_power': (self.ev_data[agent_idx]['soc'] - self.ev_data[agent_idx]['initial_soc']) * self.C_k, # kWh
                'charging_time': (self.ev_data[agent_idx]['departure_time'] - self.ev_data[agent_idx]['arrival_time']).seconds / 3600 # Convert seconds to hours
            }
            
            # Record the SoC history
            self.soc_history.loc[len(self.soc_history)] = ({
                'requestID': self.ev_data[agent_idx]['requestID'],  # Add 'requestID' to the soc_history DataFrame
                'current_time': self.ev_data[agent_idx]['departure_time'], # Add the current time to the soc_history DataFrame
                'soc': self.ev_data[agent_idx]['soc'], # Add the current SoC to the soc_history DataFrame
                'SoC_upper_bound': self.SoC_upper_bound_list[agent_idx], # Add the upper bound of SoC to the soc_history DataFrame
                'SoC_lower_bound': self.SoC_lower_bound_list[agent_idx]  # Add the lower bound of SoC to the soc_history DataFrame
            })
            
            # Reset EV data for the selected charging pile
            self.ev_data[agent_idx] = {'requestID': None, 
                                        'arrival_time': None, 
                                        'departure_time': None, 
                                        'initial_soc': 0.0, 
                                        'departure_soc': 0.0,
                                        'soc': 0.0, 
                                        'time_before_soc_max': None,
                                        'time_before_soc_min': None,
                                        'connected': False} 
            
            
        else:
            logger.bind(console=True).warning(f"Charging pile {agent_idx} is not connected.")
            return None
    
    
    """Get the current parking status and EV data for each charging pile."""
    def get_parking_status(self):
        return {
            'current_parking': self.current_parking,
            'current_parking_number': self.current_parking_number,
            'ev_data': [data for data in self.ev_data if data.get('requestID') is not None]
        }
    
    
    """Get the EV data for a specific charging pile."""
    def get_agent_status(self, agent_idx):
        if agent_idx < 0 or agent_idx >= self.num_agents:
            logger.bind(console=True).info(f"Agent index {agent_idx} out of range.")
            return None
        return self.ev_data[agent_idx]
    
    
    """Reset the environment."""
    def reset(self):
        self.ev_data = [{'requestID': None, 
                         'arrival_time': None, 
                         'departure_time': None, 
                         'initial_soc': 0.0, 
                         'departure_soc': 0.0,
                         'soc': 0.0, 
                         'time_before_soc_max': None,
                         'time_before_soc_min': None,
                         'connected': False} 
                        for _ in range(self.num_agents)]
        
        self.current_parking = np.zeros(self.num_agents, dtype=bool)
        self.current_parking_number = 0
        self.timestamp = self.start_time
        
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
        
        return self.ev_data


    """Update the SoC of a specific charging pile."""
    def step(self, agent_idx, action: float, current_time: datetime, SoC_lower_bound, SoC_upper_bound, time_interval: int = 60):
        
        # Record the SoC history
        self.soc_history.loc[len(self.soc_history)] = ({
            'requestID': self.ev_data[agent_idx]['requestID'],  # Add 'requestID' to the soc_history DataFrame
            'current_time': current_time, # Add the current time to the soc_history DataFrame
            'soc': self.ev_data[agent_idx]['soc'], # Add the current SoC to the soc_history DataFrame
            'SoC_upper_bound': self.SoC_upper_bound_list[agent_idx], # Add the upper bound of SoC to the soc_history DataFrame
            'SoC_lower_bound': self.SoC_lower_bound_list[agent_idx]  # Add the lower bound of SoC to the soc_history DataFrame
        })

        # Update the SoC of the charging pile
        self.ev_data[agent_idx]['soc'] = (self.ev_data[agent_idx]['soc'] * self.C_k + action * (time_interval / 60)) / self.C_k  # Update SoC
        self.ev_data[agent_idx]['soc'] = np.clip(self.ev_data[agent_idx]['soc'], self.soc_min, self.soc_max)  # Ensure SoC is within a reasonable range
        self.SoC_lower_bound_list[agent_idx], self.SoC_upper_bound_list[agent_idx] = SoC_lower_bound, SoC_upper_bound


    """Get the SoC of a specific charging pile."""
    def get_soc(self, agent_idx):
        return self.ev_data[agent_idx]['soc']
    
    
    """Get the maximum and minimum SoC based on the current time."""
    def get_soc_max_and_min(self, agent_idx, current_time: datetime):
        
        SoC_lower_bound, SoC_upper_bound = self.soc_min, self.soc_max # Initialize the lower and upper bounds of SoC
        current_parking_number = self.current_parking_number if self.current_parking_number > 0 else 1   # Get the number of previously connected charging piles
        
        # Calculate the time needed to reach the departure SoC
        t_needed = self.ev_data[agent_idx]['departure_time'] - \
            timedelta(minutes=int((( self.ev_data[agent_idx]['departure_soc'] - \
                self.soc_min) * self.C_k / (self.max_charging_power / current_parking_number)) * 60)) # Calculate the time needed to reach the departure SoC

        if current_time > t_needed:
            # Calculate the lower bound of SoC based on the time needed to reach the departure SoC
            SoC_lower_bound = self.soc_min + \
                (self.ev_data[agent_idx]['departure_soc'] - self.soc_min) * ((current_time - t_needed).seconds / (self.ev_data[agent_idx]['departure_time'] - t_needed).seconds)
                
            # Calculate the charging power based on the time needed to reach the departure SoC
            charging_power = (SoC_lower_bound - self.soc_min) * self.C_k / (current_time - t_needed).seconds * 3600 
            if charging_power > self.max_charging_power:
                SoC_lower_bound = self.soc_min + (self.max_charging_power / current_parking_number) * (current_time - t_needed).seconds / 3600 / self.C_k 
                
        elif current_time < self.ev_data[agent_idx]['time_before_soc_min']:
            # Calculate the upper bound of SoC based on the time before the SoC reaches the minimum value
            SoC_lower_bound = \
                self.ev_data[agent_idx]['initial_soc'] + self.max_discharging_power * (current_time - self.ev_data[agent_idx]['arrival_time']).seconds / 3600 / self.C_k
            
        if current_time < self.ev_data[agent_idx]['time_before_soc_max']:
            # Calculate the upper bound of SoC based on the time before the SoC reaches the maximum value
            SoC_upper_bound = \
                self.ev_data[agent_idx]['initial_soc'] + self.max_charging_power * (current_time - self.ev_data[agent_idx]['arrival_time']).seconds / 3600 / self.C_k
                
        # Ensure the SoC bounds are within a reasonable range
        SoC_lower_bound = np.clip(SoC_lower_bound, self.soc_min, self.soc_max)
        SoC_upper_bound = np.clip(SoC_upper_bound, self.soc_min, self.soc_max)

        return SoC_lower_bound, SoC_upper_bound


    """Get the power constraints for the DEB algorithm."""
    def get_deb_constraints(self, agent_idx, current_time: datetime):
        
        SoC_tk = self.ev_data[agent_idx]['soc'] # Get the current SoC
        SoC_lower_bound, SoC_upper_bound = self.get_soc_max_and_min(agent_idx, current_time) # Get the maximum and minimum SoC
        
        eta_star = 1 / self.eta if SoC_tk > SoC_lower_bound else self.eta # Calculate the charging efficiency based on the current SoC
        P_max_tk = min(self.max_charging_power, (SoC_upper_bound - SoC_tk) * self.C_k * eta_star) # Calculate the maximum charging power
        P_min_tk = max(self.max_discharging_power, (SoC_lower_bound - SoC_tk) * self.C_k * eta_star) #  Calculate the minimum charging power
        
        return P_max_tk, P_min_tk, SoC_lower_bound, SoC_upper_bound
    
    
    """Get the maximum and minimum charging power based on the current SoC."""
    def get_P_max_tk_and_P_min_tk(self, agent_idx, SoC_lower_bound, SoC_upper_bound):
        
        SoC_tk = self.ev_data[agent_idx]['soc'] # Get the current SoC
        eta_star = 1 / self.eta if SoC_tk > SoC_lower_bound else self.eta # Calculate the charging efficiency based on the current SoC
        P_max_tk = min(self.max_charging_power, (SoC_upper_bound - SoC_tk) * self.C_k * eta_star) # Calculate the maximum charging power
        P_min_tk = max(self.max_discharging_power, (SoC_lower_bound - SoC_tk) * self.C_k * eta_star) #  Calculate the minimum charging power
        
        return P_max_tk, P_min_tk
