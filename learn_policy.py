import os

import torch
from gym import Env
from gym.vector import AsyncVectorEnv, VectorEnv
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from factories import HalfCheetahEnvFactory, FallingEnvFactory, HumanoidFallingEnvFactory, HumanoidEnvFactory
from radam import RAdam
from utils.action_getters import ActionGetter, ActionGetterModule
from utils.dicts import ArrayKey, TensorKey, ModuleKey, ModuleDict
from utils.loss_calculators import LossCalculator, LossCalculatorInputTarget, LossCalculatorLambda
from utils.nets import ProbMLPConstantLogStd, MultiLayerPerceptron, ScalerNet
from utils.sample_collectors import SampleCollectorV0, SampleCollector
from utils.tensor_inseter import TensorInserter, TensorInserterTensorize, TensorInserterForward, TensorInserterLambda, \
    TensorInserterModuleLambda
import torch.nn as nn

from utils.trainers import Trainee, RLTrainer
from utils.module_updaters import ModuleUpdater, ModuleUpdaterOptimizer
import numpy as np

from utils.utils import EnvContainer


def main():
    n_envs = len(os.sched_getaffinity(0))
    factory = FallingEnvFactory()
    # factory = HalfCheetahEnvFactory()
    # factory = HumanoidFallingEnvFactory()
    env: Env = factory.make_env()
    envs: VectorEnv = AsyncVectorEnv([factory.make_env for _ in range(n_envs)])
    env_container = EnvContainer(env, envs)

    state_dim, = env.observation_space.shape
    action_dim, = env.action_space.shape
    relu = nn.ReLU()
    tanh = nn.Tanh()
    identity = nn.Identity()

    actor = ProbMLPConstantLogStd(state_dim, action_dim, [256, 256], relu, tanh, -1.0)
    critic = MultiLayerPerceptron(state_dim, 1, [256, 256], relu, identity)
    scaler_ = StandardScaler()
    print("Fit scaler")
    env.reset()
    state_seq = []
    for _ in tqdm(range(512)):
        action = env.action_space.sample()
        state, _, done, _ = env.step(action)
        state_seq.append(state)
        if done:
            env.reset()
    state_seq = np.stack(state_seq)
    scaler_.fit(state_seq)
    scaler = ScalerNet(scaler_)

    module_dict = ModuleDict()
    module_dict.set(ModuleKey.actor, actor)
    module_dict.set(ModuleKey.scaler, scaler)
    module_dict.set(ModuleKey.critic, critic)

    action_getter: ActionGetter = ActionGetterModule(actor, scaler)
    sample_collector: SampleCollector = SampleCollectorV0(env_container, action_getter, 2048, 1)

    mse_loss = nn.MSELoss()
    critic_tensor_inserter: TensorInserter = \
        TensorInserterTensorize(ArrayKey.states, TensorKey.states_tensor) + \
        TensorInserterTensorize(ArrayKey.log_probs, TensorKey.log_probs_tensor) + \
        TensorInserterTensorize(ArrayKey.cumulative_rewards, TensorKey.cumulative_rewards_tensor) + \
        TensorInserterForward(TensorKey.states_tensor, ModuleKey.scaler, TensorKey.states_tensor) + \
        TensorInserterForward(TensorKey.states_tensor, ModuleKey.critic, TensorKey.cumulative_reward_predictions_tensor)
    critic_loss_calculator: LossCalculator = \
        LossCalculatorInputTarget(TensorKey.cumulative_reward_predictions_tensor, TensorKey.cumulative_rewards_tensor,
                                  mse_loss)

    actor_tensor_inserter: TensorInserter = \
        TensorInserterTensorize(ArrayKey.states, TensorKey.states_tensor) + \
        TensorInserterTensorize(ArrayKey.actions, TensorKey.actions_tensor) + \
        TensorInserterTensorize(ArrayKey.log_probs, TensorKey.log_probs_tensor) + \
        TensorInserterTensorize(ArrayKey.cumulative_rewards, TensorKey.cumulative_rewards_tensor) + \
        TensorInserterForward(TensorKey.states_tensor, ModuleKey.scaler, TensorKey.states_tensor) + \
        TensorInserterForward(TensorKey.states_tensor, ModuleKey.critic,
                              TensorKey.cumulative_reward_predictions_tensor) + \
        TensorInserterLambda([TensorKey.cumulative_rewards_tensor, TensorKey.cumulative_reward_predictions_tensor],
                             lambda x, y: x - y, TensorKey.advantages_tensor) + \
        TensorInserterModuleLambda(ModuleKey.actor, [TensorKey.states_tensor, TensorKey.actions_tensor],
                                   lambda actor, state, action: actor.get_log_prob(state, action),
                                   TensorKey.new_log_probs_tensor) + \
        TensorInserterLambda([TensorKey.new_log_probs_tensor, TensorKey.log_probs_tensor, TensorKey.advantages_tensor],
                             get_ppo_surrogate_tensor, TensorKey.ppo_surrogates_tensor)

    actor_loss_calculator: LossCalculator = \
        LossCalculatorLambda([TensorKey.ppo_surrogates_tensor], lambda x: -torch.mean(x))

    actor_optimizer = RAdam(params=actor.parameters(), lr=3e-4)
    actor_updater: ModuleUpdater = ModuleUpdaterOptimizer(actor_optimizer)
    critic_optimizer = RAdam(params=critic.parameters(), lr=3e-4)
    critic_updater: ModuleUpdater = ModuleUpdaterOptimizer(critic_optimizer)

    actor_trainee = Trainee([actor], actor_updater, actor_tensor_inserter, actor_loss_calculator, 10)
    critic_trainee = Trainee([critic], critic_updater, critic_tensor_inserter, critic_loss_calculator, 10)

    trainer = RLTrainer(sample_collector, [critic_trainee, actor_trainee], 100000, 128)
    trainer.train(module_dict)


def get_ppo_surrogate_tensor(new_log_probs_tensor, old_log_probs_tensor, advantages_tensor):
    ratio = torch.exp(new_log_probs_tensor - old_log_probs_tensor)
    surr1 = ratio * advantages_tensor
    surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages_tensor
    return torch.min(surr1, surr2)


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    main()
