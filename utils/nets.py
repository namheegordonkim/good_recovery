from abc import abstractmethod
from typing import List

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.distributions import Normal


class DummyNet(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ScalerNet(nn.Module):

    def __init__(self, scaler: StandardScaler):
        super().__init__()
        self.scaler = scaler

    def forward(self, x: torch.Tensor):
        x_np = x.cpu().detach().numpy()
        if len(x.shape) == 1:
            x_np = x_np.reshape(1, -1)
        x_transformed = self.scaler.transform(x_np)
        if len(x.shape) == 1:
            x_transformed = x_transformed.reshape(-1)
        return torch.as_tensor(x_transformed).float()


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], activation: nn.Module, final_layer_activation: nn.Module):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation: nn.Module = activation
        self.final_layer_activation = final_layer_activation
        self.fcs: nn.ModuleList = nn.ModuleList()
        layer_dims: np.ndarray = np.array([input_dim, *hidden_dims, output_dim])
        for dim1, dim2 in zip(layer_dims, layer_dims[1:]):
            fc = nn.Linear(dim1, dim2)
            self.fcs.append(fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fc in self.fcs[:-1]:
            x = fc(x)
            x = self.activation.forward(x)
        x = self.fcs[-1](x)
        x = self.final_layer_activation(x)
        return x


class ProbNet(nn.Module):
    """
    Two-handed network whose forward() returns a distribution
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        raise NotImplementedError


class ProbMLPConstantLogStd(ProbNet):

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], activation: nn.Module,
                 final_layer_activation: nn.Module, log_std: float):
        super().__init__()
        self.mlp = MultiLayerPerceptron(input_dim, output_dim, hidden_dims, activation, final_layer_activation)
        self.log_std = log_std  # fixed log_std is superior for exploration

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        mu: torch.Tensor = self.mlp.forward(x)
        log_std: torch.Tensor = torch.ones_like(mu) * self.log_std
        return mu, log_std

    def get_log_prob(self, input: torch.Tensor, output: torch.Tensor):
        """
        Sampling has produced the output. Based on current distribution, what is the probability?
        """
        mu, log_std = self.forward(input)
        std = torch.exp(log_std)
        normal = Normal(mu, std)
        log_prob = normal.log_prob(output)
        return log_prob
