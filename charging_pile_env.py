import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from logger_config import configured_logger as logger

class EVChargingEnv:
    def __init__(self, num_agents):  
        self.num_agents = num_agents
        self.current_parking = np.zeros(num_agents, dtype=bool)  # Whether the charging pile is connected
        self.current_parking_number = 0  # Number of currently connected charging piles
        self.previous_parking_number = 0  # Number of previously connected charging piles
        
        # Define charging pile power constraints
        self.max_charging_power = 16  # kW
        self.max_discharging_power = -16  # kW
        self.C_k = 24  # Battery capacity in kWh
        self.eta = 0.95  # Charging efficiency
        self.soc_max = 0.9  # Maximum SoC
        self.soc_min = 0.2  # Minimum SoC

        # Initialize EV data for each charging pile
        self.ev_data = [{'requestID': None, 
                         'arrival_time': None, 
                         'departure_time': None, 
                         'initial_soc': 0.0, 
                         'soc': 0.0, 
                         'charging_history': [], 
                         'time_before_soc_max': None} for _ in range(self.num_agents)]
        
        # Initialize a DataFrame to store charging records and SoC history
        self.charging_records = pd.DataFrame(columns=['requestID', 'arrival_time', 'departure_time', 'initial_soc', 'final_soc', 'charging_power', 'charging_time']) # Initialize a DataFrame to store charging records
        self.soc_history = pd.DataFrame(columns=['requestID', 'current_time', 'soc', 'SoC_upper_bound', 'SoC_lower_bound']) # Initialize a DataFrame to store SoC history
        
    def add_ev(self, requestID, arrival_time, departure_time, initial_soc, departure_soc):
        available_piles = np.where(self.current_parking == False)[0]  # Find indices of unconnected charging piles
        if len(available_piles) > 0:
            selected_pile = np.random.choice(available_piles)  # Randomly select an unconnected charging pile
            self.current_parking[selected_pile] = True  # Connect the selected charging pile
            # self.current_parking_number += 1
            
            # Initialize EV data for the selected charging pile
            self.ev_data[selected_pile] = {
                'requestID': requestID,
                'arrival_time': arrival_time,
                'departure_time': departure_time,
                'initial_soc': initial_soc,
                'departure_soc': departure_soc,
                'soc': initial_soc,
                'charging_history': [],  # Initialize an empty list to store charging records
                'time_before_soc_max': arrival_time + timedelta(minutes=((self.soc_max - initial_soc) * self.C_k / self.max_charging_power) * 60), # Calculate the time before the SoC reaches the maximum value
                'time_before_soc_min': arrival_time + timedelta(minutes=((self.soc_min - initial_soc) * self.C_k / self.max_discharging_power) * 60), # Calculate the time before the SoC reaches the minimum value
            }
            
            # Log the EV data
            logger.info(f"requestID: {self.ev_data[selected_pile]['requestID']}")
            logger.info(f"initial_soc: {self.ev_data[selected_pile]['initial_soc']}")
            logger.info(f"departure_soc: {self.ev_data[selected_pile]['departure_soc']}")
            logger.info(f"arrival_time: {self.ev_data[selected_pile]['arrival_time']}")
            logger.info(f"departure_time: {self.ev_data[selected_pile]['departure_time']}")
            logger.info(f"time_before_soc_max: {self.ev_data[selected_pile]['time_before_soc_max']}")
            logger.info(f"time_before_soc_min: {self.ev_data[selected_pile]['time_before_soc_min']}")
            logger.info('-' * 50)
        else:
            logger.warning("No available charging piles.")
            return None
        
    def remove_ev(self, agent_idx):
        if self.current_parking[agent_idx]:
            self.current_parking[agent_idx] = False  # Disconnect the selected charging pile
            # Record the charging history
            self.charging_records.loc[len(self.charging_records)] = {
                'requestID': self.ev_data[agent_idx]['requestID'],
                'arrival_time': self.ev_data[agent_idx]['arrival_time'],
                'departure_time': self.ev_data[agent_idx]['departure_time'],
                'initial_soc': self.ev_data[agent_idx]['initial_soc'],
                'final_soc': self.ev_data[agent_idx]['soc'],
                'charging_power': (self.ev_data[agent_idx]['soc'] - self.ev_data[agent_idx]['initial_soc']) * self.C_k, # kWh
                'charging_time': (self.ev_data[agent_idx]['departure_time'] - self.ev_data[agent_idx]['arrival_time']).seconds / 3600 # Convert seconds to hours
            }
            
            # Reset EV data for the selected charging pile
            self.ev_data[agent_idx] = {'requestID': None, 
                                       'arrival_time': None, 
                                       'departure_time': None, 
                                       'initial_soc': 0.0, 
                                       'soc': 0.0, 
                                       'charging_history': [], 
                                       'time_before_soc_max': None}
        else:
            logger.warning(f"Charging pile {agent_idx} is not connected.")
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
            print(f"Agent index {agent_idx} out of range.")
            return None
        return self.ev_data[agent_idx]
    
    
    """Reset the environment."""
    def reset(self):
        self.current_step = 0
        self.ev_data = [{'requestID': None, 'arrival_time': None, 'departure_time': None, 'initial_soc': 0.0, 'soc': 0.0, 'charging_history': [], 'time_before_soc_max': None} for _ in range(self.num_agents)]
        self.current_parking = np.zeros(self.num_agents, dtype=bool)
        self.current_parking_number = 0
        return self.ev_data


    """Update the SoC of a specific charging pile."""
    def step(self, agent_idx, action: float, current_time: datetime, SoC_lower_bound, SoC_upper_bound, time_interval: int = 60):
    
        self.ev_data[agent_idx]['soc'] = (self.ev_data[agent_idx]['soc'] * self.C_k + action * (time_interval / 60)) / self.C_k  # Update SoC
        self.ev_data[agent_idx]['soc'] = np.clip(self.ev_data[agent_idx]['soc'], self.soc_min, self.soc_max)  # Ensure SoC is within a reasonable range
        
        # Record the SoC history
        self.soc_history.loc[len(self.soc_history)] = ({
            'requestID': self.ev_data[agent_idx]['requestID'],  # Add 'requestID' to the soc_history DataFrame
            'current_time': current_time, # Add the current time to the soc_history DataFrame
            'soc': self.ev_data[agent_idx]['soc'], # Add the current SoC to the soc_history DataFrame
            'SoC_upper_bound': SoC_upper_bound, # Add the upper bound of SoC to the soc_history DataFrame
            'SoC_lower_bound': SoC_lower_bound  # Add the lower bound of SoC to the soc_history DataFrame
        })


    """Get the SoC of a specific charging pile."""
    def get_soc(self, agent_idx):
        return self.ev_data[agent_idx]['soc']
    
    
    """Get the maximum and minimum SoC based on the current time."""
    def get_soc_max_and_min(self, agent_idx, current_time: datetime):
        
        SoC_lower_bound, SoC_upper_bound = self.soc_min, self.soc_max # Initialize the lower and upper bounds of SoC
        previous_parking_number = self.previous_parking_number if self.previous_parking_number > 0 else 1   # Get the number of previously connected charging piles
        
        # Calculate the time needed to reach the departure SoC
        t_needed = self.ev_data[agent_idx]['departure_time'] - \
            timedelta(minutes=int((( self.ev_data[agent_idx]['departure_soc'] - \
                self.soc_min) * self.C_k / (self.max_charging_power / previous_parking_number)) * 60)) # Calculate the time needed to reach the departure SoC

        if current_time > t_needed:
            # Calculate the lower bound of SoC based on the time needed to reach the departure SoC
            SoC_lower_bound = self.soc_min + \
                (self.ev_data[agent_idx]['departure_soc'] - self.soc_min) * ((current_time - t_needed).seconds / (self.ev_data[agent_idx]['departure_time'] - t_needed).seconds)
                
            # Calculate the charging power based on the time needed to reach the departure SoC
            charging_power = (SoC_lower_bound - self.soc_min) * self.C_k / (current_time - t_needed).seconds * 3600 
            if charging_power > self.max_charging_power:
                SoC_lower_bound = self.soc_min + (self.max_charging_power / previous_parking_number) * (current_time - t_needed).seconds / 3600 / self.C_k 
                
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

if __name__ == '__main__':

    # Set random seed for reproducibility and create an EV charging environment
    np.random.seed(42)
    num_agents = 10
    env = EVChargingEnv(num_agents)

    # Generate random EV request data
    base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    arrival_times = [base_date + timedelta(hours=int(np.random.uniform(8, 12))) for _ in range(num_agents)]
    departure_times = [base_date + timedelta(hours=int(np.random.uniform(16, 22))) for _ in range(num_agents)]
    initial_socs = np.round(np.random.uniform(0.3, 0.5, num_agents), 2)
    departure_socs = np.round(np.random.uniform(0.5, 0.9, num_agents), 2)


    ev_request_data = []
    for i in range(num_agents):
        ev_request_data.append({
            'requestID': i + 1000,  # Generate a unique request ID
            'arrival_time': arrival_times[i], # Generate a random arrival time
            'departure_time': departure_times[i], # Generate a random departure time
            'initial_soc': initial_socs[i], # Generate a random initial SoC
            'departure_soc': departure_socs[i] # Generate a random departure SoC
        })
        
    # Run the simulation
    base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) 
    current_time = base_date
    end_time = base_date + timedelta(hours=24)
    time_interval = 1  # 1 hour

    while current_time <= end_time:
        
        count = 0
        # Add EVs that arrived at the current time
        if current_time not in [ev['arrival_time'] for ev in ev_request_data]:
            logger.info(f"Current time: {current_time} | No EVs arrived.")
        else:
            for ev in ev_request_data:
                if ev['arrival_time'] == current_time:
                    env.add_ev(ev['requestID'], ev['arrival_time'], ev['departure_time'], ev['initial_soc'], ev['departure_soc'])
                    env.current_parking_number += 1
                    count += 1
                    logger.info(f"EV {ev['requestID']} arrived at {current_time}")

        # Update the SoC of each connected EV
        for agent_idx, is_parking in enumerate(env.current_parking):
            if is_parking:
                ev_data = env.get_agent_status(agent_idx)
                P_max_tk, P_min_tk, SoC_lower_bound, SoC_upper_bound = env.get_deb_constraints(agent_idx, current_time)
                action = np.random.uniform(P_min_tk, P_max_tk)  # Randomly select a charging power
                
                # Update the SoC of the connected EV
                env.step(agent_idx, action, current_time, SoC_lower_bound, SoC_upper_bound, time_interval=60)
                soc = env.get_soc(agent_idx)
                logger.info(f"EV {ev_data['requestID']} - SoC: {soc}")
        
        # Remove EVs that departed at the current time
        if current_time in [ev['departure_time'] for ev in ev_request_data]:
            for ev in ev_request_data:
                if ev['departure_time'] == current_time:
                    agent_idx = np.where([ev['requestID'] == data['requestID'] for data in env.ev_data])[0][0]
                    env.remove_ev(agent_idx)
                    env.current_parking_number -= 1
                    count -= 1
                    logger.info(f"EV {ev['requestID']} departed at {current_time}")
        
        env.previous_parking_number += count
        current_time += timedelta(hours=time_interval)

    # Save the charging records and SoC history to CSV files
    env.soc_history.to_csv('./Result/soc_history.csv', index=False)
    env.charging_records.to_csv('./Result/charging_records.csv', index=False)