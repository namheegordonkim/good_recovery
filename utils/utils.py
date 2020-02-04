from abc import abstractmethod

import torch
from gym import Env
from gym.vector import VectorEnv

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"


class EnvContainer:
    """
    Usually we will keep track of the "alpha environment" (env) and a pack of vectorized environments (envs).
    EnvContainer binds the two into member variables of one object.
    State tracking is also facilitated here, thus call EnvContainer's methods rather than VecEnv's methods.
    """

    def __init__(self, env: Env, envs: VectorEnv):
        self.env = env
        self.envs = envs
        self.env_state = env.reset()
        self.envs_states = envs.reset()
        self.n_envs = envs.num_envs

    def env_step(self, action):
        self.env_state = self.env.step(action)
        return self.env_state

    def envs_step(self, actions):
        next_states, rewards, dones, infos = self.envs.step(actions)
        self.envs_states = next_states
        return next_states, rewards, dones, infos

    def env_reset(self):
        self.env_state = self.env.reset()
        return self.env_state

    def envs_reset(self):
        self.envs_states = self.envs.reset()
        return self.envs_states


class EnvFactory:

    @abstractmethod
    def make_env(self):
        raise NotImplementedError
