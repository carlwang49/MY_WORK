import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class EVChargingEnv:
    def __init__(self, num_agents, max_charging_power=6, max_discharging_power=-6, C_k=24, eta=0.95):
        self.num_agents = num_agents
        self.max_charging_power = max_charging_power
        self.max_discharging_power = max_discharging_power
        self.C_k = C_k
        self.eta = eta
        self.evs = []

    def add_ev(self, t_arrival, t_departure, required_soc):
        ev = {
            "t_arrival": t_arrival,
            "t_departure": t_departure,
            "required_soc": required_soc,
            "current_soc": 0,
            "soc_trajectory": [],
            "P_max": [],
            "P_min": []
        }
        self.evs.append(ev)

    def simulate(self, total_time):
        time = np.arange(0, total_time, 1)  # in minutes
        for t in time:
            for ev in self.evs:
                if t >= ev["t_arrival"] and t <= ev["t_departure"]:
                    # Calculate dynamic upper and lower bounds
                    SoC_UB = min(ev["required_soc"], ev["current_soc"] + (self.max_charging_power * self.eta) / self.C_k)
                    SoC_LB = max(ev["current_soc"], ev["current_soc"] + (self.max_discharging_power * self.eta) / self.C_k)
                    
                    # Calculate maximum and minimum charging power
                    P_max = min(self.max_charging_power, (SoC_UB - ev["current_soc"]) * self.C_k / self.eta)
                    P_min = max(self.max_discharging_power, (SoC_LB - ev["current_soc"]) * self.C_k / self.eta)
                    
                    ev["soc_trajectory"].append(ev["current_soc"])
                    ev["P_max"].append(P_max)
                    ev["P_min"].append(P_min)
                    
                    # Update current SoC based on max charging power for the next step
                    ev["current_soc"] += (P_max * self.eta) / self.C_k
                else:
                    ev["soc_trajectory"].append(ev["current_soc"])
                    ev["P_max"].append(0)
                    ev["P_min"].append(0)

    def plot(self):
        time = np.arange(len(self.evs[0]["soc_trajectory"]))
        for i, ev in enumerate(self.evs):
            plt.figure(figsize=(10, 6))
            plt.plot(time, ev["soc_trajectory"], label='SoC Trajectory')
            plt.plot(time, ev["P_max"], label='Max Charging Power', linestyle='--')
            plt.plot(time, ev["P_min"], label='Min Charging Power', linestyle='--')
            plt.xlabel('Time (minutes)')
            plt.ylabel('State of Charge (SoC)')
            plt.title(f'EV {i+1} Charging/Discharging Profile')
            plt.legend()
            plt.grid(True)
            plt.show()

# Example usage
env = EVChargingEnv(num_agents=10)
env.add_ev(t_arrival=0, t_departure=180, required_soc=0.8)  # EV 1
env.add_ev(t_arrival=30, t_departure=210, required_soc=0.6)  # EV 2
# Add more EVs as needed...

# Simulate for a total of 240 minutes (4 hours)
env.simulate(total_time=240)

# Plot the results
env.plot()