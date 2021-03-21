from custom_env import DelayWrapper
from reptile import ReptileCallback

import gym
import numpy as np
import time

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
import navigation_2d


def weighted_delay(idx):
    p = [0.2, 0.2, 0.2, 0.2]
    p[idx] += 0.2
    return np.random.choice(4, 1, p=p)[0]

for env_id in ["Acrobot-v1", "Pendulum-v0"]:
    callback = ReptileCallback(8, env_id=env_id)
    env = DelayWrapper(env=gym.make(env_id), delay_fn=lambda: weighted_delay(0))
    model  = A2C(env=env, policy='MlpPolicy', verbose=0)
    model.learn(50000, callback=callback)
    model.save("reptile_a2c")

    for i in range(4):
        env = DelayWrapper(env=gym.make(env_id), delay_fn=lambda: weighted_delay(i))
        mean_rwd, std_rwd = evaluate_policy(model, env, n_eval_episodes=100)
        print(mean_rwd, std_rwd, sep=", ")

    env = DelayWrapper(env=gym.make(env_id), delay_fn=lambda: 0)
    model  = A2C(env=env, policy='MlpPolicy', verbose=0)
    model.learn(50000)
    model.save("a2c")

    for i in range(4):
        env = DelayWrapper(env=gym.make(env_id), delay_fn=lambda: weighted_delay(i))
        mean_rwd, std_rwd = evaluate_policy(model, env, n_eval_episodes=100)
        print(mean_rwd, std_rwd, sep=", ")