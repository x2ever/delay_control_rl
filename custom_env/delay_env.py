import gym


class Wrapper:
    def __init__(self, env: gym.Env):
        self._env = env
    
    def __getattr__(self, name):
        return self._env.__getattr__(name)


class DelayWrapper(Wrapper):
    def __init__(self, env: gym.Env, delay_fn):
        self._env = env
        self._delay_fn = delay_fn
    
    def __getattr__(self, name):
        return self._env.__getattr__(name)

    def step(self, action):
        delay_n: int = self._delay_fn()
        total_reward = 0
        for _ in range(delay_n + 1):
            obs, reward, done, _ = self._env.step(action)
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, {}
    
    def reset(self):
        return self._env.reset()
        

if __name__ == "__main__":
    import gym
    import time
    
    from stable_baselines3 import A2C, DQN

    env = gym.make("Navi-Acc-Lidar-Obs-Task1-v0")
    env = DelayWrapper(env=env, delay_fn=lambda: 1)
    model  = DQN(env=env, policy='MlpPolicy', verbose=1, )
    model.learn(100000)
