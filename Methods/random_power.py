from EVChargingEnv import EVChargingEnv
import numpy as np
from datetime import datetime, timedelta
from logger_config import configured_logger as logger


if __name__ == '__main__':

    # Set random seed for reproducibility and create an EV charging environment
    np.random.seed(42)
    num_agents = 20
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
    env.soc_history.to_csv('../Result/soc_history.csv', index=False)
    env.charging_records.to_csv('../Result/charging_records.csv', index=False)