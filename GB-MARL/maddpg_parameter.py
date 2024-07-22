import argparse
from EVBuildingEnvMADDPG import EVBuildingEnv
from dotenv import load_dotenv
import os
load_dotenv()

NUMBER_OF_EPISODES = int(os.getenv('NUMBER_OF_EPISODES'))
LEARN_INTERVAL = int(os.getenv('LEARN_INTERVAL'))
RANDOM_STEPS = int(float(os.getenv('RANDOM_STEPS')))
TAU = float(os.getenv('TAU'))
GAMMA = float(os.getenv('GAMMA'))
AGENT_BUFFER_CAPACITY = int(float(os.getenv('AGENT_BUFFER_CAPACITY')))
TOP_LEVEL_BUFFER_CAPACITY = int(float(os.getenv('TOP_LEVEL_BUFFER_CAPACITY')))
AGENT_BATCH_SIZE = int(os.getenv('AGENT_BATCH_SIZE'))
TOP_LEVEL_BATCH_SIZE = int(os.getenv('TOP_LEVEL_BATCH_SIZE'))
LEARNING_RATE_ACTOR = float(os.getenv('LEARNING_RATE_ACTOR'))
LEARNING_RATE_CRITIC = float(os.getenv('LEARNING_RATE_CRITIC'))
EPSILON = float(os.getenv('EPSILON'))
SIGMA = float(os.getenv('SIGMA'))
SIGMA_DECAY = float(os.getenv('SIGMA_DECAY'))

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode_num', type=int, default=NUMBER_OF_EPISODES, help='total episode num during training procedure')
    # parser.add_argument('--episode_length', type=int, default=10, help='steps per episode')
    parser.add_argument('--learn_interval', type=int, default=LEARN_INTERVAL, help='steps interval between learning time')
    parser.add_argument('--random_steps', type=int, default=RANDOM_STEPS, help='random steps before the agent start to learn')
    parser.add_argument('--tau', type=float, default=TAU, help='soft update parameter')
    parser.add_argument('--gamma', type=float, default=GAMMA, help='discount factor')
    parser.add_argument('--buffer_capacity', type=int, default=AGENT_BUFFER_CAPACITY, help='capacity of replay buffer')
    parser.add_argument('--top_level_buffer_capacity', type=int, default=TOP_LEVEL_BUFFER_CAPACITY, help='capacity of top level replay buffer')
    parser.add_argument('--batch_size', type=int, default=AGENT_BATCH_SIZE, help='batch-size of replay buffer')
    parser.add_argument('--top_level_batch_size', type=int, default=TOP_LEVEL_BATCH_SIZE, help='batch-size of top level replay buffer')
    parser.add_argument('--actor_lr', type=float, default=LEARNING_RATE_ACTOR, help='learning rate of actor')
    parser.add_argument('--critic_lr', type=float, default=LEARNING_RATE_CRITIC, help='learning rate of critic')
    parser.add_argument('--epsilon', type=float, default=EPSILON, help='epsilon-greedy')
    parser.add_argument('--sigma', type=float, default=SIGMA, help='sigma of the noise')
    parser.add_argument('--sigma_decay', type=float, default=SIGMA_DECAY, help='decay rate of sigma')
    
    return parser.parse_args()


def get_env(num_agents, start_time, end_time):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = EVBuildingEnv(num_agents, start_time, end_time)
    
    new_env.reset()
    _dim_info = {}
    _top_dim_info = []
    _top_dim_info.append(new_env.get_top_level_observation_space().shape[0])
    _top_dim_info.append(new_env.get_top_level_action_space().size())
    
    for agent_id in new_env.agents: 
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
        _dim_info[agent_id].append(new_env.action_space(agent_id).size())
    
    return new_env, _dim_info, _top_dim_info