
import gym
import navigation_2d
import random

from stable_baselines3.common.callbacks import BaseCallback
from custom_env import DelayWrapper
from copy import deepcopy
from typing import List
from collections import OrderedDict
import numpy as np


def weighted_delay(idx):
    p = [0.15, 0.15, 0.15, 0.15]
    p[idx] += 0.4
    return np.random.choice(4, 1, p=p)[0]


def aggregate_model_parameter(
    base_parameter_dict: OrderedDict,
    sub_model_parameters: List,
    alpha: float,
):
    new_parameter = OrderedDict()
    for key in base_parameter_dict.keys():
        layer_parameter = []
        for i in range(len(sub_model_parameters)):
            layer_parameter.append(sub_model_parameters[i][key])
        delta = sum(layer_parameter) / len(layer_parameter)
        new_parameter[key] = (1 - alpha) * base_parameter_dict[key] + alpha * delta
    return new_parameter


class ReptileCallback(BaseCallback):
    def __init__(self, meta_batch_size: int, env_id, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.meta_batch_size = meta_batch_size
        self.env_id = env_id

    def _init_callback(self) -> None:
        self.old_model_param = deepcopy(self.model.policy.state_dict())
        self.meta_batch = []

    def _on_step(self) -> None:
        return True

    def _on_rollout_start(self) -> None:
        self.meta_batch.append(deepcopy(self.model.policy.state_dict()))

        if len(self.meta_batch) % self.meta_batch_size == 0:
            self.reptile_update()
            self.meta_batch = []

        self.model.policy.load_state_dict(self.old_model_param)
        self.change_task(len(self.meta_batch))

    def reptile_update(self) -> None:
        new_model_param = aggregate_model_parameter(
            deepcopy(self.old_model_param),
            self.meta_batch,
            alpha=0.25,  # FIXME:
        )
        self.old_model_param = deepcopy(new_model_param)

        if self.verbose > 0:
            print("reptile update")

    def change_task(self, random_task_num) -> None:
        env = self.make_env_with_task_num(random_task_num)
        self.model.set_env(env)
        self.model.env.reset()

        if self.verbose > 0:
            print(f"change task {random_task_num}")

    def make_env_with_task_num(self, task):
        if task >= 4:
            return DelayWrapper(env=gym.make(self.env_id), delay_fn=lambda: task - 4)
        else:
            return DelayWrapper(env=gym.make(self.env_id), delay_fn=lambda: weighted_delay(task))

