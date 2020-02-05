import torch
from torch import nn

from factories import HalfCheetahEnvFactory, FallingEnvFactory, HumanoidEnvFactory, HumanoidFallingEnvFactory
from utils.action_getters import ActionGetterModule, ActionGetter
from utils.dicts import ModuleDict, ModuleKey
from utils.nets import ProbMLPConstantLogStd, ScalerNet


def main():
    factory = FallingEnvFactory()
    # factory = HalfCheetahEnvFactory()
    # factory = HumanoidFallingEnvFactory()
    module_dict: ModuleDict = torch.load("./saves/latest.pt")
    actor: ProbMLPConstantLogStd = module_dict.get(ModuleKey.actor)
    scaler: ScalerNet = module_dict.get(ModuleKey.scaler)
    action_getter: ActionGetter = ActionGetterModule(actor, scaler)
    env = factory.make_env()
    state = env.reset()
    while True:
        env.render()
        action = action_getter.get_action(state)
        # action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        print(reward)
        state = next_state
        if done:
            state = env.reset()
        import time;
        time.sleep(0.001)
        print(reward, action)


if __name__ == "__main__":
    main()
