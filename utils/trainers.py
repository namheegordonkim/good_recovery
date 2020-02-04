from abc import abstractmethod
from typing import List

import torch
from tqdm import tqdm

from utils.dicts import ModuleDict, ArrayDict, TensorDict, ArrayKey, TensorKey, ModuleKey
from utils.loss_calculators import LossCalculator
from utils.sample_collectors import SampleCollector
from utils.tensor_inseter import TensorInserter, TensorInserterTensorize, TensorInserterForward

import torch.nn as nn
import numpy as np


class Trainer:
    """
    An object with train() method.
    Its constructor should cover all the data-agnostic logical components necessary in Tensemble framework:
    1. (For RL) SampleCollector
    2. TensorInserters
    3. LossCalculators
    """

    @abstractmethod
    def train(self, module_dict: ModuleDict):
        raise NotImplementedError


class ModuleUpdater:

    @abstractmethod
    def update_module(self, loss):
        raise NotImplementedError


class ModuleUpdaterOptimizer(ModuleUpdater):

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def update_module(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Trainee:
    """
    Encapsulate target modules, tensor inserter and a loss calculator with update frequency in terms of epochs
    """

    def __init__(self, modules: List[nn.Module], module_updater: ModuleUpdater, tensor_inserter: TensorInserter,
                 loss_calculator: LossCalculator,
                 n_epochs: int):
        self.modules = modules
        self.module_updater = module_updater
        self.tensor_inserter = tensor_inserter
        self.loss_calculator = loss_calculator
        self.n_epochs = n_epochs


class RLTrainer(Trainer):

    def __init__(self, sample_collector: SampleCollector, trainees: List[Trainee], n_cycles: int, batch_size: int):
        self.sample_collector = sample_collector
        self.trainees = trainees
        self.n_cycles = n_cycles
        self.batch_size = batch_size

    def train(self, module_dict: ModuleDict):
        for i in tqdm(range(self.n_cycles)):
            self.train_one_cycle(module_dict)

            if i % 20 == 0:
                # at the end of each cycle, use the alpha environment to report total reward
                total_reward = 0
                env = self.sample_collector.env_container.env
                action_getter = self.sample_collector.action_getter
                state = env.reset()
                for _ in range(200):
                    # env.render()
                    action = action_getter.get_action(state)
                    new_state, reward, done, _ = env.step(action)
                    state = new_state
                    total_reward += reward
                    if done:
                        break
                # env.close()
                print("Cycle {:02d}\tReward:{:.2f}".format(i, total_reward))
                torch.save(module_dict, "./saves/latest.pt")
                print("Saved to ./saves/latest.pt")

    def train_one_cycle(self, module_dict: ModuleDict):
        """
        A cycle refers to the cycle through the trainees.
        By default, one sample collection is done per cycle.
        """
        array_dict: ArrayDict = self.sample_collector.collect_samples_by_number()
        n_batches = int(array_dict.n_examples / self.batch_size)
        all_idxs = np.random.choice(array_dict.n_examples, array_dict.n_examples, replace=False)
        for trainee in self.trainees:
            for epoch in range(trainee.n_epochs):
                batch_idxs = np.array_split(all_idxs, n_batches)
                tensor_dict: TensorDict = TensorDict()
                for batch_idx in batch_idxs:
                    trainee.tensor_inserter.insert_tensor(tensor_dict, array_dict, module_dict, batch_idx)
                    loss = trainee.loss_calculator.calculate_loss(tensor_dict)
                    trainee.module_updater.update_module(loss)
