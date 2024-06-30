import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from scipy.optimize import linprog

# 電價設置
prices = [0.056] * 8 + [0.092] * 4 + [0.267] * 6 + [0.092] * 5 + [0.056] * 1

# 電動車的設定
battery_capacity = 24  # kWh
initial_soc = 0.4
final_soc = 0.7
min_soc = 0.2
max_soc = 0.9

# 每個時間段的最大和最小功率設置
max_powers = [6, 6, 5, 5, 4, 4, 3, 3, 3]  # 充電功率限制
min_powers = [-3, -3, -4, -4, -5, -5, -6, -6, -6]  # 放電功率限制（負值）

# 確保 max_powers 和 min_powers 的長度等於 num_intervals
arrival_time = 8
departure_time = 17
num_intervals = departure_time - arrival_time

if len(max_powers) != num_intervals or len(min_powers) != num_intervals:
    raise ValueError("max_powers 和 min_powers 的長度必須等於 num_intervals")

# 計算進來和離開的能量需求
initial_energy = initial_soc * battery_capacity
required_energy = final_soc * battery_capacity
min_energy = min_soc * battery_capacity
max_energy = max_soc * battery_capacity

print(f'prices: {prices[arrival_time:departure_time]}')
print(f"Initial energy: {round(initial_energy, 2)} kWh")
print(f"Required energy: {round(required_energy, 2)} kWh")
print(f"Min energy: {round(min_energy, 2)} kWh")
print(f"Max energy: {round(max_energy, 2)} kWh")

# 計算需要考慮的時間範圍
time_intervals = list(range(arrival_time, departure_time))

# 取得相關的電價
relevant_prices = prices[arrival_time:departure_time]

# 定義線性規劃問題
c = np.concatenate((relevant_prices, -np.array(relevant_prices)))  # 目標函數的係數（成本）

# 確保總能量變化等於所需的能量變化
A_eq = np.concatenate((np.ones(num_intervals), -np.ones(num_intervals))).reshape(1, -1)
b_eq = np.array([required_energy - initial_energy])

# 充電和放電的功率約束
A_ub = np.zeros((2 * num_intervals + 2 * num_intervals, 2 * num_intervals))
b_ub = np.zeros(2 * num_intervals + 2 * num_intervals)

for i in range(num_intervals):
    A_ub[i, i] = 1  # 充電功率 <= max_powers[i]
    A_ub[num_intervals + i, num_intervals + i] = 1  # 放電功率 <= -min_powers[i]（注意這裡的放電功率是負值，所以要取反）
    b_ub[i] = max_powers[i]
    b_ub[num_intervals + i] = -min_powers[i]

# SoC 約束：確保 SoC 不低於 min_soc 和不超過 max_soc
for i in range(num_intervals):
    A_ub[2 * num_intervals + i, :i + 1] = 1
    A_ub[2 * num_intervals + i, num_intervals:num_intervals + i + 1] = -1
    b_ub[2 * num_intervals + i] = max_energy - initial_energy
    
    A_ub[3 * num_intervals + i, :i + 1] = -1
    A_ub[3 * num_intervals + i, num_intervals:num_intervals + i + 1] = 1
    b_ub[3 * num_intervals + i] = initial_energy - min_energy

# 變數的上下限
bounds = [(0, max_powers[i]) for i in range(num_intervals)] + [(0, -min_powers[i]) for i in range(num_intervals)]

# 解決線性規劃問題
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