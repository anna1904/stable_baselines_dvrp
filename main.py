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

model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1, tensorboard_log="./a2c_cartpole_tensorboard/")
model.learn(total_timesteps=1000000, log_interval=4)
model.save("dvrp_v0")

obs = env.reset()

for i in range(10):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

































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

