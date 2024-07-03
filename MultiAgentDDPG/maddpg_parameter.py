import argparse
from EVBuildingEnvMADDPG import EVBuildingEnv

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode_num', type=int, default=3000, help='total episode num during training procedure')
    parser.add_argument('--episode_length', type=int, default=25, help='steps per episode')
    parser.add_argument('--learn_interval', type=int, default=100, help='steps interval between learning time')
    parser.add_argument('--random_steps', type=int, default=5e4, help='random steps before the agent start to learn')
    parser.add_argument('--tau', type=float, default=0.02, help='soft update parameter')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='capacity of replay buffer')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch-size of replay buffer')
    parser.add_argument('--actor_lr', type=float, default=0.0001, help='learning rate of actor')
    parser.add_argument('--critic_lr', type=float, default=0.001, help='learning rate of critic')
    
    return parser.parse_args()


def get_env(num_agents, start_time, end_time):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = EVBuildingEnv(num_agents, start_time, end_time)
    
    new_env.reset()
    _dim_info = {}
    for agent_id in new_env.agents: 
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
        _dim_info[agent_id].append(new_env.action_space(agent_id).size())

    return new_env, _dim_info