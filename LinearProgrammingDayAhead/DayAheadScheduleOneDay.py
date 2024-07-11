import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from scipy.optimize import linprog

# set the electricity prices
prices = [0.056] * 8 + [0.092] * 4 + [0.267] * 6 + [0.092] * 5 + [0.056] * 1

# ev settings
battery_capacity = 24  # kWh
initial_soc = 0.4
final_soc = 0.7
min_soc = 0.2
max_soc = 0.9

# max and min power settings for each time interval
max_powers = [6, 6, 5, 5, 4, 4, 3, 3, 3]  # max charging power
min_powers = [-3, -3, -4, -4, -5, -5, -6, -6, -6]  # max discharging power (negative)

# ensure that the lengths of max_powers and min_powers are equal to num_intervals
arrival_time = 8
departure_time = 17
num_intervals = departure_time - arrival_time

if len(max_powers) != num_intervals or len(min_powers) != num_intervals:
    raise ValueError("max_powers 和 min_powers 的長度必須等於 num_intervals")

# calculate the energy requirements
initial_energy = initial_soc * battery_capacity
required_energy = final_soc * battery_capacity
min_energy = min_soc * battery_capacity
max_energy = max_soc * battery_capacity

print(f'prices: {prices[arrival_time:departure_time]}')
print(f"Initial energy: {round(initial_energy, 2)} kWh")
print(f"Required energy: {round(required_energy, 2)} kWh")
print(f"Min energy: {round(min_energy, 2)} kWh")
print(f"Max energy: {round(max_energy, 2)} kWh")

# calculate the time intervals
time_intervals = list(range(arrival_time, departure_time))

# get the relevant electricity prices
relevant_prices = prices[arrival_time:departure_time]

# define the linear programming problem
c = np.concatenate((relevant_prices, -np.array(relevant_prices)))  # coefficients of the objective function (cost)

# ensure that the total energy change is equal to the required energy change
A_eq = np.concatenate((np.ones(num_intervals), -np.ones(num_intervals))).reshape(1, -1)
b_eq = np.array([required_energy - initial_energy])

# charge and discharge power constraints
A_ub = np.zeros((2 * num_intervals + 2 * num_intervals, 2 * num_intervals))
b_ub = np.zeros(2 * num_intervals + 2 * num_intervals)

for i in range(num_intervals):
    A_ub[i, i] = 1  # charging power <= max_powers[i]
    A_ub[num_intervals + i, num_intervals + i] = 1  # discharging power <= -min_powers[i] (note that the discharging power is negative, so we need to take the negative)
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

# variable upper and lower limits
bounds = [(0, max_powers[i]) for i in range(num_intervals)] + [(0, -min_powers[i]) for i in range(num_intervals)]

# solve the linear programming problem
res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method='highs')

if res.success:
    schedule = []
    current_energy = initial_energy
    total_cost = 0
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
        
        current_energy += energy if action == 'charge' else -energy
        current_soc = current_energy / battery_capacity
        total_cost += cost
        schedule.append({
            'time': time_intervals[i],
            'action': action,
            'energy': round(energy, 2),
            'current_energy': round(current_energy, 2),
            'current_soc': round(current_soc, 2),
            'cost': round(cost, 2),
            'total_cost': round(total_cost, 2)
        })
    
    print(f"Total cost: {round(total_cost, 2)}")
    print("Schedule:")
    for entry in schedule:
        print(entry)
else:
    raise ValueError("Charging demand cannot be met within the given time frame.")