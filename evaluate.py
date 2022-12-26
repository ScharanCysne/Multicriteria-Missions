import sys
import pygame
import warnings
import numpy as np

from interface                          import Interface
from constants                          import *
from environment                        import CoverageMissionEnv
from stable_baselines3                  import PPO
from pettingzoo.test                    import parallel_api_test
from utils                              import plot_results

warnings.filterwarnings("ignore")

NUM_DRONES = 15
NUM_OBSTACLES = 100
NUM_EPISODES = 200
NUM_TIMESTEPS = 900

# Load Model
#model = PPO.load(f"tmp/model_15_10000")
#model = PPO.load(f"output/policy_20_2000")
#model = PPO.load(f"model_b_3")
#model = PPO.load(f"model_b_4")
#model = PPO.load(f"model_b_5")
#model = PPO.load(f"model_b_10")
model = PPO.load(f"model_b_15")
#model = PPO.load(f"model_b_20")
#model = PPO.load(f"model_b_20_attacks")

# Creation of Environment
env = CoverageMissionEnv(NUM_OBSTACLES, NUM_DRONES, EVALUATION)
parallel_api_test(env, num_cycles=1000)

# Render interface
#interface = Interface()
episode_time = []
timesteps = 0
record = False

algebraic_connectivity = np.zeros((NUM_EPISODES, NUM_TIMESTEPS))
robustness_level = np.zeros((NUM_EPISODES, NUM_TIMESTEPS))
area_coverage = np.zeros((NUM_EPISODES, NUM_TIMESTEPS))

for episode in range(NUM_EPISODES):
    print(episode)
    # Init episode
    timesteps = 0
    done = False
    obs = env.reset(scenario=episode+1)
    while not done and timesteps < NUM_TIMESTEPS:
        timesteps += 1
        actions = dict()
        for agent in obs:
            actions[agent] = model.predict(obs[agent], deterministic=True)[0]
        obs, rewards, dones, infos = env.step(actions)
        
        if dones[agent]:
            done = True
        algebraic_connectivity[episode, timesteps-1] = env.env_state.algebraic_connectivity
        robustness_level[episode, timesteps-1] = env.env_state.network_robustness
        area_coverage[episode, timesteps-1] = env.env_state.network_coverage
        
    env.close()
    record = False

# Plot Algebraic Connectivity, Area Coverage, Robustness Level
plot_results(NUM_DRONES, NUM_TIMESTEPS, robustness_level, algebraic_connectivity, area_coverage)