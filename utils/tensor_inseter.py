from abc import abstractmethod
from typing import List

import torch
import numpy as np

from utils.dicts import TensorDict, ArrayDict, TensorKey, ArrayKey, ModuleDict, ModuleKey


class TensorInserter:
    """
    An object that transfers tensors and arrays into a TensorDict.
    Defines the operation in a data-agnostic way through initializing constructors with keys.
    Make sure the only way to put numpy arrays into TensorDict is through an inserter.
    """

    @abstractmethod
    def insert_tensor(self, tensor_dict: TensorDict, array_dict: ArrayDict, module_dict: ModuleDict,
                      batch_idx: np.ndarray):
        raise NotImplementedError

    def __add__(self, other):
        """
        Compose two TensorInserters into a sequence of operation (left -> right)
        """
        return TensorInserterPair(self, other)


class TensorInserterPair(TensorInserter):

    def __init__(self, left: TensorInserter, right: TensorInserter):
        self.left = left
        self.right = right

    def insert_tensor(self, tensor_dict: TensorDict, array_dict: ArrayDict, module_dict: ModuleDict,
                      batch_idx: np.ndarray):
        tensor_dict = self.left.insert_tensor(tensor_dict, array_dict, module_dict, batch_idx)
        tensor_dict = self.right.insert_tensor(tensor_dict, array_dict, module_dict, batch_idx)
        return tensor_dict


class TensorInserterTensorize(TensorInserter):
    """
    Transfer numpy arrays to torch tensors. Make sure the number of dimensions is at least 2
    """

    def __init__(self, array_key: ArrayKey, tensor_key: TensorKey, dtype=torch.float):
        self.array_key = array_key
        self.tensor_key = tensor_key
        self.dtype = dtype

    def insert_tensor(self, tensor_dict: TensorDict, array_dict: ArrayDict, module_dict: ModuleDict,
                      batch_idx: np.ndarray):
        array = array_dict.get(self.array_key)[batch_idx]
        tensor = torch.as_tensor(array, dtype=self.dtype)
        if len(tensor.shape) == 1:
            tensor = tensor.reshape(-1, 1)
        tensor_dict.set(self.tensor_key, tensor)
        return tensor_dict


class TensorInserterForward(TensorInserter):

    def __init__(self, source_key: TensorKey, module_key: ModuleKey, target_key: TensorKey):
        self.source_key = source_key
        self.module_key = module_key
        self.target_key = target_key

    def insert_tensor(self, tensor_dict: TensorDict, array_dict: ArrayDict, module_dict: ModuleDict,
                      batch_idx: np.ndarray):
        source_tensor = tensor_dict.get(self.source_key)
        module = module_dict.get(self.module_key)
        target_tensor = module.forward(source_tensor)
        tensor_dict.set(self.target_key, target_tensor)
        return tensor_dict


class TensorInserterLambda(TensorInserter):
    """
    Allow function pointer input for custom transformations
    """

    def __init__(self, source_keys: List[TensorKey], f, target_key: TensorKey):
        self.source_keys = source_keys
        self.f = f
        self.target_key = target_key

    def insert_tensor(self, tensor_dict: TensorDict, array_dict: ArrayDict, module_dict: ModuleDict,
                      batch_idx: np.ndarray):
        input_tensors = [tensor_dict.get(k) for k in self.source_keys]
        output_tensor = self.f(*input_tensors)
        tensor_dict.set(self.target_key, output_tensor)
        return tensor_dict


class TensorInserterModuleLambda(TensorInserter):
    """
    Allow function pointer input for custom transformations
    """

    def __init__(self, module_key: ModuleKey, source_keys: List[TensorKey], f, target_key: TensorKey):
        self.module_key = module_key
        self.source_keys = source_keys
        self.f = f
        self.target_key = target_key

    def insert_tensor(self, tensor_dict: TensorDict, array_dict: ArrayDict, module_dict: ModuleDict,
                      batch_idx: np.ndarray):
        module = module_dict.get(self.module_key)
        input_tensors = [tensor_dict.get(k) for k in self.source_keys]
        output_tensor = self.f(module, *input_tensors)
        tensor_dict.set(self.target_key, output_tensor)
        return tensor_dict
