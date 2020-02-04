from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActionGetter:
    """
    Generic object that can serve as a policy.
    State-based action getter. Assume stochastic actors.
    Implements get_action() and sample_action() methods.
    """

    @abstractmethod
    def get_action(self, state) -> np.ndarray:
        """
        Take single instance of state as 1D numpy array and output corresponding action
        """
        raise NotImplementedError

    @abstractmethod
    def sample_action(self, state) -> (np.ndarray, float):
        """
        Take single instance of state as 1D numpy array, get corresponding mu, sigma.
        Based on mu, sigma, sample an action. Report the action and the log_prob of getting the action..
        Make sure log_prob is differentiable.
        """
        raise NotImplementedError


class ActionGetterModule(ActionGetter):

    def __init__(self, actor: nn.Module, scaler: nn.Module, activation_function: nn.Module):
        self.actor = actor
        self.scaler = scaler
        self.activation_function = activation_function

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Take states as n by d numpy array and output corresponding actions
        """
        state_tensor = torch.as_tensor(state).float()
        if len(state.shape) == 1:
            state_tensor = state_tensor.reshape(1, -1)
        state_scaled_tensor = self.scaler.forward(state_tensor)
        action_tensor, _ = self.actor.forward(state_scaled_tensor)
        action_activated_tensor = self.activation_function.forward(action_tensor)
        action = action_activated_tensor.detach().numpy()
        if len(state.shape) == 1:
            action = action.reshape(-1)
        return action

    def sample_action(self, state) -> (np.ndarray, float):
        state_tensor = torch.as_tensor(state).float()
        if len(state.shape) == 1:
            state_tensor = state_tensor.reshape(1, -1)
        state_scaled_tensor = self.scaler.forward(state_tensor)
        mu_tensor, log_sigma_tensor = self.actor.forward(state_scaled_tensor)
        sigma_tensor = torch.exp(log_sigma_tensor)
        normal_distribution = Normal(mu_tensor, sigma_tensor)
        action_tensor = normal_distribution.sample()
        log_prob_tensor = normal_distribution.log_prob(action_tensor)
        action = action_tensor.detach().numpy()
        log_prob = log_prob_tensor.detach().numpy()
        if len(state.shape) == 1:
            action = action.reshape(-1)
            log_prob = log_prob.reshape(-1)

        return action, log_prob
