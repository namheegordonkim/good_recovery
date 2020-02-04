from abc import abstractmethod

import numpy as np

from utils.action_getters import ActionGetter
from utils.dicts import ArrayDict, ArrayKey
from utils.utils import EnvContainer


class SampleCollector:
    """
    Object with methods for collecting experience samples for an agent.
    User must provide an environment to draw samples from.
    User must provide an "action getter".
    """

    def __init__(self, env_container: EnvContainer, action_getter: ActionGetter, n_samples: int, horizon: int):
        self.env_container = env_container
        self.action_getter = action_getter
        self.n_samples = n_samples
        self.horizon = horizon
        if self.horizon <= 0:
            raise RuntimeError("horizon must be strictly positive")
        if self.n_samples <= 0:
            raise RuntimeError("n_samples must be strictly positive")

    @abstractmethod
    def collect_samples_by_number(self) -> ArrayDict:
        """
        Collect samples until the specified number of experience tuples are collected.
        """
        raise NotImplementedError

    @abstractmethod
    def collect_samples_by_horizon(self) -> ArrayDict:
        """
        Collect samples until the specified horizon is reached.
        """
        raise NotImplementedError


class SampleCollectorV0(SampleCollector):

    def collect_samples_by_horizon(self) -> ArrayDict:
        states_seq = []
        actions_seq = []
        log_probs_seq = []
        next_states_seq = []
        rewards_seq = []
        dones_seq = []
        for _ in range(self.horizon):
            states = self.env_container.envs_states
            actions, log_probs = self.action_getter.sample_action(states)
            next_states, rewards, dones, _ = self.env_container.envs_step(actions)

            states_seq.append(states)
            actions_seq.append(actions)
            log_probs_seq.append(log_probs)
            next_states_seq.append(next_states)
            rewards_seq.append(rewards)
            dones_seq.append(dones)

        array_dict: ArrayDict = ArrayDict(self.env_container.n_envs * self.horizon)
        for key, value in zip(
                [ArrayKey.states, ArrayKey.actions, ArrayKey.log_probs, ArrayKey.actions,
                 ArrayKey.next_states, ArrayKey.rewards, ArrayKey.dones],
                [states_seq, actions_seq, log_probs_seq, next_states_seq, rewards_seq, dones_seq]):
            array_dict.set(key, np.concatenate(value, axis=0))
        return array_dict

    def collect_samples_by_number(self):
        """
        Collect experience until gathered tuples exceed specified number.
        Reset after every horizon.
        """
        state_dim, = self.env_container.env.observation_space.shape
        action_dim, = self.env_container.env.action_space.shape

        states_seq = []
        actions_seq = []
        log_probs_seq = []
        next_states_seq = []
        rewards_seq = []
        dones_seq = []
        n_samples_collected = 0
        while n_samples_collected < self.n_samples:

            self.env_container.envs_reset()

            for _ in range(self.horizon):
                states = self.env_container.envs_states
                actions, log_probs = self.action_getter.sample_action(states)
                next_states, rewards, dones, _ = self.env_container.envs_step(actions)

                states_seq.append(states)
                actions_seq.append(actions)
                log_probs_seq.append(log_probs)
                next_states_seq.append(next_states)
                rewards_seq.append(rewards)
                dones_seq.append(dones)

                n_samples_collected += self.env_container.n_envs
                if n_samples_collected >= self.n_samples:
                    break

        states_matrix = np.stack(states_seq, axis=1)
        actions_matrix = np.stack(actions_seq, axis=1)
        log_probs_matrix = np.stack(log_probs_seq, axis=1)
        rewards_matrix = np.stack(rewards_seq, axis=1)
        next_states_matrix = np.stack(next_states_seq, axis=1)
        dones_matrix = np.stack(dones_seq, axis=1)

        # cumulative rewards
        cumulative_rewards_matrix = compute_cumulative_rewards_mat(rewards_matrix, dones_matrix, 0.99)

        array_dict: ArrayDict = ArrayDict(n_samples_collected)
        array_dict.set(ArrayKey.states, states_matrix.reshape(-1, state_dim))
        array_dict.set(ArrayKey.actions, actions_matrix.reshape(-1, action_dim))
        array_dict.set(ArrayKey.log_probs, log_probs_matrix.reshape(-1, action_dim))
        array_dict.set(ArrayKey.rewards, rewards_matrix.reshape(-1))
        array_dict.set(ArrayKey.dones, dones_matrix.reshape(-1))
        array_dict.set(ArrayKey.next_states, next_states_matrix.reshape(-1, state_dim))
        array_dict.set(ArrayKey.cumulative_rewards, cumulative_rewards_matrix.reshape(-1))

        return array_dict


def compute_cumulative_rewards(rewards: np.ndarray, dones: np.ndarray, gamma: float):
    """
    Delimited by done=True, compute cumulative rewards for a single trajectory
    """
    done_idxs, = np.where(dones)
    done_idxs = np.concatenate([[-1], done_idxs]).astype(np.int)
    cumulative_rewards = np.zeros_like(rewards)
    for i, j in zip(done_idxs, done_idxs[1:]):
        rewards_sub = rewards[i + 1:j + 1]
        cumulative_rewards_sub = np.zeros_like(rewards_sub)
        for k in np.arange(len(rewards_sub)-2, -1, -1):
            cumulative_rewards_sub[k] = rewards_sub[k] + gamma * cumulative_rewards_sub[k+1]
        cumulative_rewards[i + 1:j + 1] = cumulative_rewards_sub

    return cumulative_rewards


def compute_cumulative_rewards_mat(rewards: np.ndarray, dones: np.ndarray, gamma: float):
    cumulative_rewards = np.zeros_like(rewards)
    for i in range(cumulative_rewards.shape[0]):
        cumulative_rewards[i] = compute_cumulative_rewards(rewards[i], dones[i], gamma)
    return cumulative_rewards
