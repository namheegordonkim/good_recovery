from abc import abstractmethod
from typing import List

from utils.dicts import TensorDict, TensorKey

import torch.nn as nn


class LossCalculator:
    """
    An object that calculates differentiable loss function based on specified tensor keys.
    Handle both RL loss and other neural network losses
    """

    @abstractmethod
    def calculate_loss(self, tensor_dict: TensorDict):
        raise NotImplementedError

    def __add__(self, other):
        return LossCalculatorAddition(self, other)


class LossCalculatorAddition(LossCalculator):

    def __init__(self, left: LossCalculator, right: LossCalculator):
        self.left = left
        self.right = right

    def calculate_loss(self, tensor_dict: TensorDict):
        left_loss = self.left.calculate_loss(tensor_dict)
        right_loss = self.right.calculate_loss(tensor_dict)
        return left_loss + right_loss


class LossCalculatorInputTarget(LossCalculator):

    def __init__(self, input_key: TensorKey, target_key: TensorKey, loss_module: nn.Module, weight=1.):
        self.input_key = input_key
        self.target_key = target_key
        self.loss_module = loss_module
        self.weight = weight

    def calculate_loss(self, tensor_dict: TensorDict):
        input_tensor = tensor_dict.get(self.input_key)
        target_tensor = tensor_dict.get(self.target_key)
        loss = self.weight * self.loss_module.forward(input_tensor, target_tensor)
        return loss


class LossCalculatorLambda(LossCalculator):

    def __init__(self, input_keys: List[TensorKey], f):
        self.f = f
        self.input_keys = input_keys

    def calculate_loss(self, tensor_dict: TensorDict):
        input_tensors = [tensor_dict.get(k) for k in self.input_keys]
        loss = self.f(*input_tensors)
        return loss

