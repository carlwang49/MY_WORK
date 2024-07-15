def dp_decide_group(observations):
    n = len(observations)
    charge_dp = [0] * (n + 1)
    discharge_dp = [0] * (n + 1)
    
    for i, obs in enumerate(observations.values(), 1):
        soc = obs['soc']
        current_price = obs['current_price']
        building_load = obs['building_load']
        
        charge_need = 0.4 * (100 - soc) + 0.3 * current_price + 0.3 * building_load
        discharge_need = 0.4 * soc + 0.3 * current_price + 0.3 * building_load
        
        if soc < 50:
            charge_dp[i] = charge_dp[i - 1] + charge_need
            discharge_dp[i] = discharge_dp[i - 1]
        else:
            charge_dp[i] = charge_dp[i - 1]
            discharge_dp[i] = discharge_dp[i - 1] + discharge_need
    
    if charge_dp[n] > discharge_dp[n]:
        return "charge"
    else:
        return "discharge"

# 示例状态信息
observations = {
    "agent_A": {"soc": 70, "current_price": 0.15, "building_load": 0.6, "emergency": 0.2},
    "agent_B": {"soc": 50, "current_price": 0.14, "building_load": 0.5, "emergency": 0.5},
    "agent_C": {"soc": 30, "current_price": 0.16, "building_load": 0.7, "emergency": 0.8}
}

# 计算当前组别
group = dp_decide_group(observations)
print(f"Current Group: {group}")