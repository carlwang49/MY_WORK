import numpy as np
import pandas as pd

class EVChargingEnv:
    def __init__(self, num_agents, max_steps=1440):  # 默认一天有1440分钟
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.current_step = 0

        # 初始化电动车状态
        self.ev_data = self.initialize_ev_data(num_agents)

        # 定义充电桩功率限制
        self.max_charging_power = 6  # kW
        self.max_discharging_power = -6  # kW
        self.C_k = 50  # 电池容量 kWh
        self.eta = 0.95  # 充电效率

        # 用于记录每小时的SoC
        self.soc_history = []

    def initialize_ev_data(self, num_agents):
        ev_data = {
            'arrival_time': np.random.randint(0, 1440, num_agents),
            'departure_time': np.random.randint(720, 1440, num_agents),
            'initial_soc': np.random.uniform(0.2, 0.6, num_agents),
            'soc': np.zeros(num_agents)
        }
        ev_data['soc'][:] = ev_data['initial_soc']
        return ev_data

    def reset(self):
        self.current_step = 0
        self.ev_data = self.initialize_ev_data(self.num_agents)
        self.soc_history = []
        return self.ev_data

    def step(self, actions: list):
        for i in range(self.num_agents):
            action = actions[i]
            P_max_tk, P_min_tk = self.get_deb_constraints(i)
            if action > P_max_tk:
                action = P_max_tk
            elif action < P_min_tk:
                action = P_min_tk   
            self.ev_data['soc'][i] += action * (1 / 60)  # 假设每步为一分钟，充电功率单位为kW
            self.ev_data['soc'][i] = np.clip(self.ev_data['soc'][i], 0.2, 0.9)  # 确保SoC在合理范围内

        if self.current_step % 60 == 0:  # 每小时记录一次
            self.soc_history.append(self.ev_data['soc'].copy())

        self.current_step += 1
        if self.current_step >= self.max_steps:
            return self.ev_data, True  # done = True

        return self.ev_data, False  # done = False

    def set_charging_power(self, agent_idx, power):
        P_max_tk, P_min_tk = self.get_deb_constraints(agent_idx)
        if power > P_max_tk:
            power = P_max_tk
        elif power < P_min_tk:
            power = P_min_tk
        self.ev_data['soc'][agent_idx] += power * (1 / 60)  # 每步为一分钟，充电功率单位为kW
        self.ev_data['soc'][agent_idx] = np.clip(self.ev_data['soc'][agent_idx], 0.2, 0.9)  # 确保SoC在合理范围内

    def get_soc(self, agent_idx):
        return self.ev_data['soc'][agent_idx]

    def get_deb_constraints(self, agent_idx):
        SoC_tk = self.ev_data['soc'][agent_idx]
        SoC_min = 0.2
        SoC_max = 0.9
        eta_star = self.eta if SoC_tk > SoC_min else 1 / self.eta
        P_max_tk = min(self.max_charging_power, (SoC_max - SoC_tk) * self.C_k * eta_star)
        P_min_tk = max(self.max_discharging_power, (SoC_min - SoC_tk) * self.C_k * eta_star)
        return P_max_tk, P_min_tk

# 示例用法
env = EVChargingEnv(num_agents=10)

# 重置环境
initial_state = env.reset()
print("Initial State:", initial_state)

# 设置充电功率并获取SoC
env.set_charging_power(agent_idx=0, power=3)  # 为第0个充电桩设置3kW的充电功率
current_soc = env.get_soc(agent_idx=0)
print("Current SoC of agent 0:", current_soc)

# 进行若干步并获取状态
for _ in range(60):  # 模拟一个小时
    new_state, done = env.step([3] * 10)  # 为所有充电桩设置3kW的充电功率

print("New State:", new_state)
print("SoC History:", env.soc_history)