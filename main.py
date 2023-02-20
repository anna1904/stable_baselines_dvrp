import stable_baselines3
import gym
from stable_baselines3.common.env_util import make_vec_env
from gym.envs.registration import register
# from sb3_contrib.common.wrappers import ActionMasker
import numpy as np
# from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
# from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from stable_baselines3 import PPO
# from stable_baselines3.common.evaluation import evaluate_policy

#
register(
    id='DVRPEnv-v0',
    entry_point='dvrp_env:DVRPEnv', #your_env_folder.envs:NameOfYourEnv
)

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.valid_action_mask()


env = gym.make("DVRPEnv-v0")
env = ActionMasker(env, mask_fn)  # Wrap to enable masking

# path = "./a2c_cartpole_tensorboard/"
# model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1, tensorboard_log=path)
# model.learn(total_timesteps=10000000, log_interval=100) #removed 2 0
# model.save("dvrp_v7")

#dvrp_v5 basic with time windows and order time in state
#dvrp_v6 I have added this missed reward for missed order
#dvrp_v7 I will give reward 1 for rejection

#
model = MaskablePPO.load("dvrp_v7", env = env)
obs = env.reset()
#
#
for i in range(480):
    action_masks = mask_fn(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    # if (action == 0):
    #     print("QQQQQ")
    obs, rewards, dones, info = env.step(action)
    env.render()

print('__________________')

obs = env.reset()
for i in range(480):
    action_masks = mask_fn(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    # if (action == 0):
    #     print("QQQQQ")
    obs, rewards, dones, info = env.step(action)
    env.render()
print('__________________')

obs = env.reset()
for i in range(480):
    action_masks = mask_fn(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    # if (action == 0):
    #     print("QQQQQ")
    obs, rewards, dones, info = env.step(action)
    env.render()
print('__________________')


































# def mask_fn(env: gym.Env) -> np.ndarray:
#     # Do whatever you'd like in this function to return the action mask
#     # for the current env. In this example, we assume the env has a
#     # helpful method we can rely on.
#     return env.valid_action_mask()
# env = ActionMasker(env, mask_fn)  # Wrap to enable masking

# model = PPO("MultiInputPolicy", env, verbose=1)
# model.learn(total_timesteps=100, log_interval=1)
# model.save("dqn_cartpole")
#
# del model
# model = PPO.load("dqn_cartpole")


# for i in range(10):
#     action, _states = model.predict(obs)
#     print('action', action)
#     obs, rewards, dones, info = env.step(action)
#     print('obs', obs)

# del model
#
# model = DQN.load("dqn_cartpole")
# obs = env.reset()
# for i in range(100):
#     action, _state = model.predict(obs, deterministic=True, action_masks=env.valid_action_mask())
#     print(action)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     # VecEnv resets automatically
#     if done:
#       obs = env.reset()

