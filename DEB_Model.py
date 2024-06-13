import numpy as np
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv('/mnt/data/complete_learning_hyperparameters_with_comments.env')

# Fetch parameters from .env file
BATTERY_CAPACITY = float(os.getenv("BATTERY_CAPACITY"))
MAX_CHARGING_POWER = float(os.getenv("MAX_CHARGING_POWER"))
SOC_MIN = float(os.getenv("SOC_MIN"))
SOC_MAX = float(os.getenv("SOC_MAX"))
N = int(os.getenv("PILES", 10))  # Default to 10 if not specified
ENERGY_CONVERSION_EFFICIENCY = float(os.getenv("ENERGY_CONVERSION_EFFICIENCY"))
DELTA_T = 1  # 時間槽間隔為1小時

class DEBModel:
    def __init__(self):
        self.max_pile_power = MAX_CHARGING_POWER
        self.min_soc = SOC_MIN
        self.max_soc = SOC_MAX
        self.station_max_power = MAX_CHARGING_POWER * N
        self.battery_capacity = BATTERY_CAPACITY
        self.efficiency = ENERGY_CONVERSION_EFFICIENCY

    def compute_needed_time(self, soc_target, t_d):
        required_time = t_d - ((soc_target - self.min_soc) * self.battery_capacity) / (self.efficiency * (self.station_max_power / N) * DELTA_T)
        return required_time

    def compute_power_bounds(self, soc_current, soc_previous):
        P_max_t = min(self.max_pile_power, (soc_current - soc_previous) * self.battery_capacity * self.efficiency)
        P_min_t = max(-self.max_pile_power, (soc_current - soc_previous) * self.battery_capacity * self.efficiency)
        return P_max_t, P_min_t

    def compute_dynamic_lower_bound(self, num_connected_ev, prev_max_power):
        P_max_d = self.station_max_power / num_connected_ev
        return min(P_max_d, prev_max_power)

# Initialize DEB Model
deb_model = DEBModel()

# Example data
soc_target = 0.8
t_d = 10  # 假設離開時間為第10個時間槽

needed_time = deb_model.compute_needed_time(soc_target, t_d)
print(f"Needed Time: {needed_time}")

soc_current = 0.5
soc_previous = 0.4

P_max_t, P_min_t = deb_model.compute_power_bounds(soc_current, soc_previous)
print(f"P_max_t: {P_max_t}, P_min_t: {P_min_t}")

num_connected_ev = 5
prev_max_power = 10

dynamic_lower_bound = deb_model.compute_dynamic_lower_bound(num_connected_ev, prev_max_power)
print(f"Dynamic Lower Bound: {dynamic_lower_bound}")