import numpy as np
import pandas as pd

# Parameters from the table
num_samples = 100  # Define the number of samples you want to generate

# Generate random samples based on the distributions and boundaries
arrival_times = np.random.normal(9, 1, num_samples)
arrival_times = np.clip(arrival_times, 7, 11)

departure_times = np.random.normal(19, 1, num_samples)
departure_times = np.clip(departure_times, 17, 21)

soc_at_arrive = np.random.normal(0.4, 0.12, num_samples)
soc_at_arrive = np.clip(soc_at_arrive, 0.2, 0.6)

# Generate random parking durations and calculate leave times
parking_durations = np.random.uniform(1, 8, num_samples)  # Assume parking durations between 1 and 8 hours
leave_times = arrival_times + parking_durations
leave_times = np.clip(leave_times, 7, 21)  # Ensure leave times are within bounds

# Generate desired SoC at leave, which should generally be higher than the initial SoC
desired_soc_at_leave = soc_at_arrive + np.random.uniform(0.2, 0.4, num_samples)
desired_soc_at_leave = np.clip(desired_soc_at_leave, 0.4, 1.0)  # Ensure it doesn't exceed 1.0

# Create a DataFrame including desired SoC at leave
data = {
    'Arrival Time': arrival_times,
    'Departure Time': departure_times,
    'Leave Time': leave_times,
    'SoC at Arrive': soc_at_arrive,
    'Desired SoC at Leave': desired_soc_at_leave
}

df = pd.DataFrame(data)

# Save to CSV
csv_path = './Data/parking_behavior.csv'
df.to_csv(csv_path, index=False)
