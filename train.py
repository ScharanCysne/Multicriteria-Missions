import os
import warnings
import supersuit as ss

from rewards                                    import save_rewards
from callback                                   import Callback
from constants                                  import *
from environment                                import CoverageMissionEnv
from stable_baselines3                          import PPO
from stable_baselines3.common.policies          import ActorCriticPolicy

warnings.filterwarnings("ignore")

# Training Parameters
NUM_DRONES = 3
NUM_OBSTACLES = 100
NUM_EPISODES = 1000
TOTAL_TIMESTEPS = NUM_EPISODES * TIMESTEPS_PER_EPISODE * 20

print(" ----------------------------------------- ")
print("Number of Agents: " + str(NUM_DRONES))
print("Number of Obstacles: " + str(NUM_OBSTACLES))
print("Number of Episodes: " + str(NUM_EPISODES))
print("Number of Timesteps per Episode: " + str(TIMESTEPS_PER_EPISODE))
print("Number of Total Timesteps: " + str(NUM_EPISODES * TIMESTEPS_PER_EPISODE * 20))
print(" ----------------------------------------- ")

# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

# Creation of Environment
env = CoverageMissionEnv(NUM_OBSTACLES, NUM_DRONES, TRAINING)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 1, num_cpus=8, base_class='stable_baselines3')

# Callback function
suffix = str(NUM_DRONES) + "_" + str(NUM_EPISODES)
callback = Callback(check_freq=TIMESTEPS_PER_EPISODE, log_dir=log_dir, suffix=suffix)

# Creation of PPO Multi-Agent model
model = PPO(
    ActorCriticPolicy,
    env,
    verbose=1,
    device="cuda",
    n_steps=TIMESTEPS_PER_EPISODE,
    batch_size=60,
    learning_rate=0.0003/NUM_DRONES,
    policy_kwargs={'net_arch': [dict(pi=[32, 32, 16], vf=[32, 32, 16])]}
)

#model = PPO.load(f"model_b_2")
#model.set_env(env)

model = model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
model.save(f"output/policy_{NUM_DRONES}_{NUM_EPISODES}")
save_rewards(log_dir, NUM_DRONES, NUM_EPISODES)