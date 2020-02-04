from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
from gym import Env
from gym.vector import VectorEnv
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from radam import RAdam
from utils.action_getters import ActionGetter, ActionGetterModule
from utils.dicts import ArrayDict, ArrayKey, TensorKey, ModuleDict, TensorDict, ModuleKey
from utils.loss_calculators import LossCalculator, LossCalculatorInputTarget, LossCalculatorLambda
from utils.nets import MultiLayerPerceptron, ProbMLPConstantLogStd, DummyNet, ScalerNet
from utils.sample_collectors import SampleCollectorV0, SampleCollector, compute_cumulative_rewards, \
    compute_cumulative_rewards_mat
from utils.tensor_inseter import TensorInserter, TensorInserterTensorize, TensorInserterForward, TensorInserterLambda
from utils.utils import EnvContainer

N_EXAMPLES = 128
N_ENVS = 32
INPUT_DIM = 17
HIDDEN_DIMS = [256, 256]
OUTPUT_DIM = 8
ACTIVATION_FUNCTION = nn.ReLU()
LOG_STD = 0.7
N_ITERS = 1000

STATE_DIM = 17
ACTION_DIM = 6

seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def create_net() -> nn.Module:
    net: nn.Module = MultiLayerPerceptron(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIMS, ACTIVATION_FUNCTION)
    return net


class TestNets(TestCase):

    def setUp(self) -> None:
        self.dummy_features: torch.Tensor = torch.rand([N_EXAMPLES, INPUT_DIM]).float()
        self.dummy_coeffs: torch.Tensor = torch.rand([INPUT_DIM, OUTPUT_DIM]).float()
        self.dummy_target: torch.Tensor = torch.matmul(self.dummy_features, self.dummy_coeffs).float()

    def test_dummy_success(self) -> None:
        net: nn.Module = DummyNet()
        dummy_output = net.forward(self.dummy_features)
        np.testing.assert_array_almost_equal(dummy_output, self.dummy_features)

    def test_scaler_success(self) -> None:
        scaler = StandardScaler()
        scaler.fit(self.dummy_features)
        net: nn.Module = ScalerNet(scaler)
        dummy_features_scaled = scaler.transform(self.dummy_features)

        dummy_example = self.dummy_features[0, :]
        dummy_example_scaled = dummy_features_scaled[0, :]
        dummy_output = net.forward(dummy_example)
        self.assertEqual(len(dummy_output.shape), 1, "Scaler output shape for 1D case is inconsistent")
        np.testing.assert_array_almost_equal(dummy_output, dummy_example_scaled)

        dummy_output = net.forward(self.dummy_features)
        self.assertEqual(len(dummy_output.shape), 2, "Scaler output shape for 2D case is inconsistent")
        np.testing.assert_array_almost_equal(dummy_output, dummy_features_scaled)

    def test_mlp_initialization(self) -> None:
        net: nn.Module = create_net()
        self.assertTrue(True, "MLP initialized with error.")

    def test_mlp_forward(self) -> None:
        net: nn.Module = create_net()
        dummy_output = net.forward(self.dummy_features)
        self.assertEqual(N_EXAMPLES, dummy_output.shape[0], "MLP output shape is inconsistent")
        self.assertEqual(OUTPUT_DIM, dummy_output.shape[1], "MLP output shape is inconsistent")

    def test_mlp_performance_linear(self) -> None:
        net: nn.Module = create_net()
        optimizer: torch.optim = RAdam(net.parameters(), lr=3e-4)
        loss_function: nn.Module = nn.MSELoss()

        # train the network
        for _ in tqdm(range(N_ITERS)):
            dummy_output: torch.Tensor = net.forward(self.dummy_features)
            loss: nn.Module = loss_function.forward(dummy_output, self.dummy_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # see the error is below threshold
        self.assertLessEqual(loss.data, 1e-2, "MLP training error for linear data is too high")

    def test_prob_mlp_initialization(self) -> None:
        net: nn.Module = ProbMLPConstantLogStd(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIMS, ACTIVATION_FUNCTION, LOG_STD)
        self.assertTrue(True, "ProbMLP initialized with error.")

    def test_prob_mlp_forward(self) -> None:
        net: nn.Module = ProbMLPConstantLogStd(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIMS, ACTIVATION_FUNCTION, LOG_STD)
        dummy_output = net.forward(self.dummy_features)
        self.assertEqual(len(dummy_output), 2, "ProbMLP output should be two-handed.")
        self.assertEqual(dummy_output[0].shape, (N_EXAMPLES, OUTPUT_DIM), "ProbMLP mu output shape is inconsistent")
        self.assertEqual(dummy_output[1].shape, (N_EXAMPLES, OUTPUT_DIM), "ProbMLP sigma output shape is inconsistent")


class TestDicts(TestCase):
    def test_array_dicts_set_success(self) -> None:
        array_dict: ArrayDict = ArrayDict(N_EXAMPLES)
        dummy_states = np.random.random((N_EXAMPLES, STATE_DIM))
        array_dict.set(ArrayKey.states, dummy_states)

        self.assertTrue(ArrayKey.states in array_dict.dict.keys())
        np.testing.assert_array_equal(array_dict.get(ArrayKey.states), dummy_states)

    def test_array_dicts_set_fail_n_examples_check(self) -> None:
        array_dict: ArrayDict = ArrayDict(N_EXAMPLES)
        dummy_states = np.random.random((N_EXAMPLES + 1, STATE_DIM))
        with self.assertRaises(RuntimeError):
            array_dict.set(ArrayKey.states, dummy_states)

    def test_array_dicts_set_fail_key_check(self) -> None:
        array_dict: ArrayDict = ArrayDict(N_EXAMPLES)
        dummy_states = np.random.random((N_EXAMPLES, STATE_DIM))
        with self.assertRaises(RuntimeError):
            array_dict.set(TensorKey.states_tensor, dummy_states)


class TestActionGetters(TestCase):
    def setUp(self) -> None:
        self.dummy_states: torch.Tensor = torch.rand([N_EXAMPLES, STATE_DIM]).float()

    def test_module_action_getter_success(self):
        actor: nn.Module = ProbMLPConstantLogStd(STATE_DIM, ACTION_DIM, HIDDEN_DIMS, ACTIVATION_FUNCTION, LOG_STD)
        scaler: nn.Module = DummyNet()
        action_getter: ActionGetter = ActionGetterModule(actor, scaler)

        dummy_state = self.dummy_states[0, :]
        dummy_action = action_getter.get_action(dummy_state)
        self.assertEqual(len(dummy_action.shape), 1, "1D case output shape is not 1D")

        dummy_actions = action_getter.get_action(self.dummy_states)
        self.assertEqual(len(dummy_actions.shape), 2, "2D case output shape is not 2D")
        self.assertTupleEqual(dummy_actions.shape, (N_EXAMPLES, ACTION_DIM), "2D case output shape is inconsistent")


class DummyVectorEnv(VectorEnv):
    def reset_wait(self, **kwargs):
        pass

    def step_wait(self, **kwargs):
        pass

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs


@patch('gym.vector.VectorEnv.step')
@patch('gym.vector.VectorEnv.reset')
@patch('gym.Env.step')
@patch('gym.Env.reset')
class TestSampleCollector(TestCase):

    def setUp(self) -> None:
        self.dummy_state = np.random.random(STATE_DIM)
        self.dummy_states = np.random.random([N_ENVS, STATE_DIM])
        self.dummy_next_states = np.random.random([N_ENVS, STATE_DIM])
        self.dummy_reward = 1
        self.dummy_rewards = np.ones(N_ENVS, dtype=np.float)
        self.dummy_done = False
        self.dummy_dones = np.zeros(N_ENVS, dtype=np.bool)
        self.dummy_info = {}

    def test_dummy_env_return_values(self, mock_env_reset, mock_env_step, mock_envs_reset, mock_envs_step) -> None:
        dummy_env = Env()
        mock_env_reset.return_value = self.dummy_state
        mock_env_step.return_value = (self.dummy_state, self.dummy_reward, self.dummy_done, self.dummy_info)
        dummy_env.reset = mock_env_reset
        dummy_env.step = mock_env_step

        dummy_env.reset()
        mock_env_reset.assert_called_with()
        dummy_action = np.random.random(ACTION_DIM)
        dummy_env.step(dummy_action)
        mock_env_step.assert_called_with(dummy_action)

    def test_dummy_vecenv_return_values(self, mock_env_reset, mock_env_step, mock_envs_reset, mock_envs_step) -> None:
        dummy_envs = DummyVectorEnv(N_ENVS, STATE_DIM, ACTION_DIM)
        mock_envs_reset.return_value = self.dummy_states
        mock_envs_step.return_value = (self.dummy_next_states, self.dummy_rewards, self.dummy_dones, {})
        dummy_envs.reset = mock_envs_reset
        dummy_envs.step = mock_envs_step

        dummy_envs.reset()
        mock_envs_reset.assert_called_with()
        dummy_actions = np.random.random([N_EXAMPLES, ACTION_DIM])
        dummy_envs.step(dummy_actions)
        mock_envs_step.assert_called_with(dummy_actions)

    def test_sample_collector_by_horizon_success(self, mock_env_reset, mock_env_step, mock_envs_reset,
                                                 mock_envs_step) -> None:
        dummy_env = Env()
        mock_env_reset.return_value = self.dummy_state
        mock_env_step.return_value = (self.dummy_state, self.dummy_reward, self.dummy_done, self.dummy_info)
        dummy_env.reset = mock_env_reset
        dummy_env.step = mock_env_step

        dummy_envs = DummyVectorEnv(N_ENVS, STATE_DIM, ACTION_DIM)
        mock_envs_reset.return_value = self.dummy_states
        mock_envs_step.return_value = (self.dummy_next_states, self.dummy_rewards, self.dummy_dones, {})
        dummy_envs.reset = mock_envs_reset
        dummy_envs.step = mock_envs_step

        dummy_env_container = EnvContainer(dummy_env, dummy_envs)
        mock_envs_reset.assert_called_once_with()  # __init__ of EnvContainer calls reset

        actor: nn.Module = ProbMLPConstantLogStd(STATE_DIM, ACTION_DIM, HIDDEN_DIMS, ACTIVATION_FUNCTION, LOG_STD)
        scaler: nn.Module = DummyNet()
        action_getter: ActionGetter = ActionGetterModule(actor, scaler)
        sample_collector: SampleCollector = SampleCollectorV0(dummy_env_container, action_getter, N_ENVS * 10, 10)

        array_dict: ArrayDict = sample_collector.collect_samples_by_horizon()
        self.assertEqual(mock_envs_step.call_count, 10)

        collected_states = array_dict.get(ArrayKey.states)
        self.assertTupleEqual(collected_states.shape, (N_ENVS * 10, STATE_DIM))

    def test_sample_collector_by_number_success(self, mock_env_reset, mock_env_step, mock_envs_reset,
                                                mock_envs_step) -> None:
        dummy_env = Env()
        mock_env_reset.return_value = self.dummy_state
        mock_env_step.return_value = (self.dummy_state, self.dummy_reward, self.dummy_done, self.dummy_info)
        dummy_env.reset = mock_env_reset
        dummy_env.step = mock_env_step

        dummy_envs = DummyVectorEnv(N_ENVS, STATE_DIM, ACTION_DIM)
        mock_envs_reset.return_value = self.dummy_states
        mock_envs_step.return_value = (self.dummy_next_states, self.dummy_rewards, self.dummy_dones, {})
        dummy_envs.reset = mock_envs_reset
        dummy_envs.step = mock_envs_step

        dummy_env_container = EnvContainer(dummy_env, dummy_envs)
        mock_envs_reset.assert_called_once_with()  # __init__ of EnvContainer calls reset

        actor: nn.Module = ProbMLPConstantLogStd(STATE_DIM, ACTION_DIM, HIDDEN_DIMS, ACTIVATION_FUNCTION, LOG_STD)
        scaler: nn.Module = DummyNet()
        action_getter: ActionGetter = ActionGetterModule(actor, scaler)
        sample_collector: SampleCollector = SampleCollectorV0(dummy_env_container, action_getter, N_ENVS * 10, 1)

        array_dict: ArrayDict = sample_collector.collect_samples_by_number()
        self.assertEqual(mock_envs_reset.call_count, 11)
        self.assertEqual(mock_envs_step.call_count, 10)

        collected_states = array_dict.get(ArrayKey.states)
        self.assertTupleEqual(collected_states.shape, (N_ENVS * 10, STATE_DIM))

    def test_cumulative_rewards_no_discount_success(self, mock_env_reset, mock_env_step, mock_envs_reset,
                                                    mock_envs_step):
        dones = np.array([0, 0, 1, 0, 0, 0, 1])
        rewards = np.array([1, 1, 1, 1, 1, 1, 1]).astype(np.float)
        ans = np.array([2, 1, 0, 3, 2, 1, 0])
        cumulative_rewards = compute_cumulative_rewards(rewards, dones, 1)

        np.testing.assert_array_equal(cumulative_rewards, ans)

    def test_cumulative_rewards_yes_discount_success(self, mock_env_reset, mock_env_step, mock_envs_reset,
                                                     mock_envs_step):
        dones = np.array([0, 0, 1, 0, 0, 0, 1])
        rewards = np.array([1, 1, 1, 1, 1, 1, 1]).astype(np.float)
        ans = np.array([1 + 1. / 2, 1, 0, 1 + 1. / 2 * (1 + 1. / 2), 1 + 1. / 2, 1, 0])
        cumulative_rewards = compute_cumulative_rewards(rewards, dones, 0.5)

        np.testing.assert_array_equal(cumulative_rewards, ans)

    def test_cumulative_rewards_mat_success(self, mock_env_reset, mock_env_step, mock_envs_reset,
                                            mock_envs_step):
        dones = np.array([
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 1, 0]
        ])
        rewards = np.array([
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1]
        ])
        ans = np.array([
            [2, 1, 0, 3, 2, 1, 0],
            [3, 2, 1, 0, 1, 0, 0]
        ])
        cumulative_rewards = compute_cumulative_rewards_mat(rewards, dones, 1)

        np.testing.assert_array_equal(cumulative_rewards, ans)


class TestTensorInserter(TestCase):

    def setUp(self) -> None:
        self.dummy_states = np.random.random([N_EXAMPLES, STATE_DIM])
        self.dummy_states_tensor = torch.as_tensor(self.dummy_states).float()
        self.dummy_module_dict = ModuleDict()

    @patch("utils.dicts.TensorDict.set")
    @patch("utils.dicts.TensorDict.get")
    @patch("utils.dicts.ArrayDict.get")
    def test_tensorize_success(self, array_dict_get, tensor_dict_get, tensor_dict_set):
        # mock
        array_dict: ArrayDict = ArrayDict(N_EXAMPLES)
        array_dict_get.return_value = self.dummy_states
        array_dict.get = array_dict_get
        tensor_dict: TensorDict = TensorDict()
        tensor_dict_get.return_value = self.dummy_states_tensor
        tensor_dict_set.return_value = None
        tensor_dict.get = tensor_dict_get
        tensor_dict.set = tensor_dict_set

        # run
        tensor_inserter: TensorInserter = TensorInserterTensorize(ArrayKey.states, TensorKey.states_tensor, torch.float)
        tensor_dict = tensor_inserter.insert_tensor(tensor_dict, array_dict, self.dummy_module_dict, np.arange(N_EXAMPLES))

        # assert
        array_dict_get.assert_called_once_with(ArrayKey.states)
        tensor_dict_set.assert_called_once()

        np.testing.assert_array_almost_equal(tensor_dict.get(TensorKey.states_tensor), self.dummy_states)

    @patch("utils.dicts.TensorDict.set")
    @patch("utils.dicts.TensorDict.get")
    @patch("utils.dicts.ModuleDict.get")
    @patch("torch.nn.Module.forward")
    def test_forward_success(self, forward_mock, module_dict_get, tensor_dict_get, tensor_dict_set):
        # mock
        net: nn.Module = DummyNet()
        forward_mock.return_value = self.dummy_states_tensor
        net.forward = forward_mock

        module_dict = ModuleDict()
        module_dict_get.return_value = net
        module_dict.get = module_dict_get

        tensor_dict = TensorDict()
        tensor_dict_get.return_value = self.dummy_states_tensor
        tensor_dict_set.return_value = None
        tensor_dict.get = tensor_dict_get
        tensor_dict.set = tensor_dict_set

        array_dict = ArrayDict(0)

        # run
        tensor_inserter = TensorInserterForward(TensorKey.states_tensor, ModuleKey.scaler, TensorKey.next_states_tensor)
        tensor_inserter.insert_tensor(tensor_dict, array_dict, module_dict, np.arange(N_EXAMPLES))

        # assert
        tensor_dict_get.assert_called_once_with(TensorKey.states_tensor)
        module_dict_get.assert_called_once_with(ModuleKey.scaler)
        forward_mock.assert_called_once_with(self.dummy_states_tensor)
        tensor_dict_set.assert_called_once()

    @patch("utils.dicts.TensorDict.set")
    @patch("utils.dicts.TensorDict.get")
    @patch("utils.dicts.ModuleDict.get")
    @patch("utils.dicts.ArrayDict.get")
    @patch("torch.nn.Module.forward")
    def test_composition_success(self, forward_mock, array_dict_get, module_dict_get, tensor_dict_get, tensor_dict_set):
        # mock
        array_dict = ArrayDict(N_EXAMPLES)
        array_dict_get.return_value = self.dummy_states
        array_dict.get = array_dict_get

        tensor_dict = TensorDict()
        tensor_dict_get.return_value = self.dummy_states_tensor
        tensor_dict_set.return_value = None
        tensor_dict.get = tensor_dict_get
        tensor_dict.set = tensor_dict_set

        net: nn.Module = DummyNet()
        forward_mock.return_value = self.dummy_states_tensor
        net.forward = forward_mock

        module_dict = ModuleDict()
        module_dict_get.return_value = net
        module_dict.get = module_dict_get

        # run
        tensor_inserter1 = TensorInserterTensorize(ArrayKey.states, TensorKey.states_tensor)
        tensor_inserter2 = TensorInserterForward(TensorKey.states_tensor, ModuleKey.scaler, TensorKey.states_tensor)
        tensor_inserter = tensor_inserter1 + tensor_inserter2
        tensor_inserter.insert_tensor(tensor_dict, array_dict, module_dict, np.arange(N_EXAMPLES))

        # assert
        array_dict_get.called_once_with(ArrayKey.states)
        self.assertEqual(tensor_dict_set.call_count, 2)
        self.assertEqual(tensor_dict_get.call_count, 1)
        tensor_dict_get.assert_called_once_with(TensorKey.states_tensor)
        module_dict_get.assert_called_once_with(ModuleKey.scaler)

    @patch("utils.dicts.TensorDict.set")
    @patch("utils.dicts.TensorDict.get")
    def test_lambda_success(self, get_mock, set_mock):
        # mock
        array_dict = ArrayDict(N_EXAMPLES)
        module_dict = ModuleDict()
        tensor_dict = TensorDict()
        get_mock.return_value = 1
        tensor_dict.get = get_mock
        tensor_dict.set = set_mock

        # run
        tensor_inserter: TensorInserter = TensorInserterLambda([TensorKey.states_tensor, TensorKey.actions_tensor],
                                                               lambda x, y: x + y, TensorKey.dones_tensor)
        tensor_inserter.insert_tensor(tensor_dict, array_dict, module_dict, np.arange(N_EXAMPLES))

        # assert
        get_mock.assert_any_call(TensorKey.states_tensor)
        get_mock.assert_any_call(TensorKey.actions_tensor)
        set_mock.assert_called_with(TensorKey.dones_tensor, 2)


class TestLossCalculator(TestCase):

    @patch("utils.dicts.TensorDict.get")
    @patch("torch.nn.MSELoss.forward")
    def test_mse_success(self, forward_mock, get_mock):
        # mock
        mse: nn.Module = nn.MSELoss()
        forward_mock.return_value = torch.as_tensor(0).float()
        mse.forward = forward_mock

        tensor_dict: TensorDict = TensorDict()
        tensor_dict.get = get_mock

        loss_calculator: LossCalculator = LossCalculatorInputTarget(TensorKey.states_tensor,
                                                                    TensorKey.next_states_tensor, mse)

        # run
        loss_calculator.calculate_loss(tensor_dict)

        # assert
        self.assertEqual(get_mock.call_count, 2)
        get_mock.assert_any_call(TensorKey.next_states_tensor)
        get_mock.assert_any_call(TensorKey.states_tensor)
        forward_mock.assert_called_once()

    @patch("torch.nn.MSELoss.forward")
    @patch("utils.dicts.TensorDict.get")
    def test_composition_success(self, get_mock, forward_mock):
        # mock
        mse: nn.Module = nn.MSELoss()
        forward_mock.return_value = torch.as_tensor(0).float()
        mse.forward = forward_mock

        tensor_dict: TensorDict = TensorDict()
        tensor_dict.get = get_mock

        loss_calculator1: LossCalculator = LossCalculatorInputTarget(TensorKey.states_tensor,
                                                                     TensorKey.next_states_tensor, mse)
        loss_calculator2: LossCalculator = LossCalculatorInputTarget(TensorKey.actions_tensor,
                                                                     TensorKey.dones_tensor, mse)
        loss_calculator = loss_calculator1 + loss_calculator2
        # run
        loss_calculator.calculate_loss(tensor_dict)

        # assert
        self.assertEqual(get_mock.call_count, 4)
        get_mock.assert_any_call(TensorKey.next_states_tensor)
        get_mock.assert_any_call(TensorKey.states_tensor)
        get_mock.assert_any_call(TensorKey.actions_tensor)
        get_mock.assert_any_call(TensorKey.dones_tensor)
        self.assertEqual(forward_mock.call_count, 2)

    @patch("utils.dicts.TensorDict.get")
    def test_lambda_success(self, get_mock):
        # mock
        tensor_dict = TensorDict()
        get_mock.return_value = 1
        tensor_dict.get = get_mock

        # run
        loss_calculator: LossCalculator = LossCalculatorLambda([TensorKey.states_tensor, TensorKey.next_states_tensor],
                                                               lambda x, y: x + y)
        loss = loss_calculator.calculate_loss(tensor_dict)

        # assert
        self.assertEqual(get_mock.call_count, 2)
        get_mock.assert_any_call(TensorKey.states_tensor)
        get_mock.assert_any_call(TensorKey.next_states_tensor)
        self.assertEqual(loss, 2)


class TestTrainers(TestCase):
    pass
