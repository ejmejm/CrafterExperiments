from multiprocessing.dummy import freeze_support
import os

import gym
from gym import Wrapper
import crafter
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import *


class AchievementInfoWrapper(gym.Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        info['achievement_count'] = np.sum(np.array(list(
            info['achievements'].values())) > 0)
        info['achievement_frac'] = info['achievement_count'] / len(info['achievements'])
        for key in info ['achievements']:
            info[key] = info['achievements'][key]
        return observation, reward, done, info

LOG_DIR = 'tmp/'
os.makedirs(LOG_DIR, exist_ok=True)


def make_env(env_name='CrafterReward-v1', data_dir='data/', n_envs=2):
    env = gym.make(env_name)
    env = AchievementInfoWrapper(env)
    env.reset()
    info_sample = env.step(0)[3]
    env = SubprocVecEnv([lambda: env for i in range(n_envs)])
    env = VecMonitor(env, LOG_DIR, info_keywords=(
        'achievement_count', 'achievement_frac') + \
        tuple(info_sample['achievements'].keys()))
    return env

class TensorboardCallback(BaseCallback):
    def __init__(self, log_dir=LOG_DIR, verbose=0):
        self.log_dir = log_dir
        self.df_idx = 0
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        df = load_results(self.log_dir)
        n_rows = df.shape[0]
        df = df.iloc[self.df_idx:n_rows]
        df.drop(columns=['index', 'r', 't', 'l'], inplace=True)
        for key in df.columns:
            if 'achievement' in key:
                val = np.mean(df[key].values)
            else:
                val = np.mean(df[key].values > 0)
            self.logger.record('achievement/' + key, val)
        self.df_idx = n_rows

env_name = 'CrafterReward-v1'

n_steps = int(2e5)

if __name__ == '__main__':
    env = make_env(env_name)