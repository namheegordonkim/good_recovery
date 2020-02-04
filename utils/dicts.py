from enum import Enum

import numpy as np
import torch
import torch.nn as nn


class Key(Enum):
    """
    Enums to introduce rigidity in dictionary access
    """
    pass


class ArrayKey(Key):
    """
    Dictionary keys solely to be used by DataDict
    """
    states = 0
    actions = 1
    next_states = 2
    rewards = 3
    log_probs = 4
    dones = 5
    cumulative_rewards = 6


class TensorKey(Key):
    """
    Dictionary keys solely to be used by TensorDict
    """
    states_tensor = 0
    actions_tensor = 1
    rewards_tensor = 2
    next_states_tensor = 3
    log_probs_tensor = 4
    dones_tensor = 5
    cumulative_reward_predictions_tensor = 6
    ppo_surrogates_tensor = 7
    cumulative_rewards_tensor = 8
    advantages_tensor = 9
    new_log_probs_tensor = 10

class ModuleKey(Key):
    """
    Dictionary keys solely to be used by ModuleDict
    """
    scaler = 0
    actor = 1
    critic = 2


class TypedDict:
    """
    Dictionary wrapper with extra type-checking.
    Use the enum Key objects.
    """

    def __init__(self):
        self.dict = dict()

    def get(self, key: Key):
        return self.dict[key]

    def set(self, key: Key, value: any) -> None:
        self.dict[key] = value


class ArrayDict(TypedDict):
    """
    TypedDict that handles numpy arrays.
    Assume that the array is at least 2 dimensional.
    Assume that the first dimension is always the number.
    """

    def __init__(self, n_examples: int):
        super().__init__()
        self.n_examples = n_examples
        self.key_class = ArrayKey

    def set(self, key: ArrayKey, value: np.ndarray):
        if key not in [k for k in self.key_class]:
            raise RuntimeError("key {} is not one of ArrayKey enums")

        if value.shape[0] != self.n_examples:
            raise RuntimeError("value has shape {} but n_examples for this ArrayDict is {}"
                               .format(value.shape, self.n_examples))

        super().set(key, value)


class TensorDict(TypedDict):
    """
        TypedDict that handles torch tensors.
        Assume that the tensor is at least 2 dimensional.
        Assume that the first dimension is always the number.
        """

    def __init__(self):
        super().__init__()
        self.key_class = TensorKey

    def set(self, key: TensorKey, value: torch.Tensor):
        if key not in [k for k in self.key_class]:
            raise RuntimeError("key {} is not one of TensorKey enums")

        super().set(key, value)


class ModuleDict(TypedDict):
    def __init__(self):
        super().__init__()
        self.key_class = ModuleKey

    def set(self, key: ModuleKey, value: nn.Module):
        if key not in [k for k in self.key_class]:
            raise RuntimeError("key {} is not one of ModuleKey enums")

        super().set(key, value)
