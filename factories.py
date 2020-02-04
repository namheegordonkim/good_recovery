import gym

from utils.utils import EnvFactory


class FallingEnvFactory(EnvFactory):

    def make_env(self):
        return gym.make("Walker2dFalling-v0")


class CartPoleEnvFactory(EnvFactory):

    def make_env(self):
        return gym.make("CartPole-v0")


class LunarLanderEnvFactory(EnvFactory):

    def make_env(self):
        return gym.make("LunarLanderContinuous-v2")


class ReacherFactory(EnvFactory):

    def make_env(self):
        return gym.make("Reacher-v2")


class HalfCheetahEnvFactory(EnvFactory):

    def make_env(self):
        return gym.make("HalfCheetah-v3")