import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from MADDPG import MADDPG
from main import get_env

# TODO: 合併 start_time, start_date, end_time, end_date

# Define the start and end date of the EV request data
start_date = START_DATE = '2018-07-01'
end_date = END_DATE = '2018-07-02'

# Define the start and end time of the EV request data
start_time = START_TIME = datetime(2018, 7, 1)
end_time = END_TIME = datetime(2018, 7, 2)

# Define the number of agents
num_agents = NUM_AGENTS = 10
parking_data_path = PARKING_DATA_PATH = '../Dataset/Sim_Parking/ev_parking_data_from_2018-07-01_to_2018-12-31.csv'



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str, default='EVBuildingEnv', help='name of the env',
                        choices=['MADDPG', 'RandomPower', 'EVBuildingEnv'])
    parser.add_argument('folder', type=str, help='name of the folder where model is saved')
    parser.add_argument('--episode-num', type=int, default=10, help='total episode num during evaluation')
    parser.add_argument('--episode-length', type=int, default=50, help='steps per episode')

    args = parser.parse_args()

    model_dir = os.path.join('../Result', args.env_name, args.folder)
    print(model_dir)
    assert os.path.exists(model_dir)

    env, dim_info = get_env(num_agents, start_time, end_time) 
    maddpg = MADDPG.load(dim_info, os.path.join(model_dir, 'model.pt'))

    agent_num = env.num_agents
    # reward of each episode of each agent
    episode_rewards = {agent: np.zeros(args.episode_num) for agent in env.agents}
    for episode in range(args.episode_num):
        states = env.reset()
        agent_reward = {agent: 0 for agent in env.agents}  # agent reward of the current episode
        while env.agents:  # interact with the env for an episode
            actions = maddpg.select_action(states)
            next_states, rewards, dones, infos = env.step(actions)
            states = next_states

            for agent_id, reward in rewards.items():  # update reward
                agent_reward[agent_id] += reward

        message = f'episode {episode + 1}, '
        # episode finishes, record reward
        for agent_id, reward in agent_reward.items():
            episode_rewards[agent_id][episode] = reward
            message += f'{agent_id}: {reward:>4f}; '
        print(message)


    # training finishes, plot reward
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    for agent_id, rewards in episode_rewards.items():
        ax.plot(x, rewards, label=agent_id)
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    total_files = len([file for file in os.listdir(model_dir)])
    title = f'evaluate result of maddpg solve {args.env_name} {total_files - 3}'
    ax.set_title(title)
    plt.savefig(os.path.join(model_dir, title))
