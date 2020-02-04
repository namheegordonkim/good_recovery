from abc import abstractmethod


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
