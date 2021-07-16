import gym
import os.path
from defeat_zerglings_banelings_env import DZBEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from absl import flags

FLAGS = flags.FLAGS
FLAGS([''])

# create vectorized environment
env = DZBEnv()
#env = DummyVecEnv([lambda: DZBEnv()])

# use ppo to learn and save the model when finished
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="log/")
if os.path.exists("model/dbz_ppo"):
    model.load("model/dbz_ppo")
model.learn(total_timesteps=int(1e5), tb_log_name="first_run", reset_num_timesteps=False)
model.save("model/dbz_ppo")