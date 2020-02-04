import gym
import numpy as np


def main():
    env = gym.make("Walker2dFalling-v0")
    env.reset()
    while True:
        env.render()
        _, _, done, _ = env.step(np.zeros(env.action_space.shape))
        if done:
            env.reset()

        import time;
        time.sleep(0.001)

if __name__ == "__main__":
    main()
