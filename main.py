import numpy.random
import stable_baselines3
import gym
import torch
from stable_baselines3.common.env_util import make_vec_env
from gym.envs.registration import register
# from sb3_contrib.common.wrappers import ActionMasker
import numpy as np
# from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
# from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.utils import set_random_seed
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


#dvrp_v5 basic with time windows and order time in state
#dvrp v8 fixed bug with rejection, no penalty PPO16
#dvrp_v_0 time step for acceptance is not inceased, state without noise PPO-3
#dvrp_v_1 returned back boolean value in state for state PPO-4
#dvrp_v_2 without acceptance decision in state, n = 10, PPO-5
#dvrp_v_3 without acceptance decision in state, n = 7, PPO-6
#dvrp_v_4 no reward for acceptance, full reward for delivering, n = 5, PPO-7
#dvrp_v_5 no reward for acceptance, full reward for delivering, without penalty for arriving empty in depot n = 5, PPO-8
#dvrp_v_6 acceptance order separately, rewards like in v_o n = 5, PPO-9
# ([self.vehicle_x_max, self.vehicle_y_max] +[self.vehicle_x_max] * self.n_orders +[self.vehicle_y_max] * self.n_orders+
# [2] * self.n_orders + reward_per_order_max + [self.vehicle_x_max, self.vehicle_y_max] +[max(self.order_reward_max)] +
# o_time_max + [self.driver_capacity] + [self.clock_max])
#dvrp_v_7 acceptance order separately,no info about new order inside orders, rewards like in v_o n = 5, PPO-10
# [self.dr_x] + [self.dr_y] + o_x + o_y + reward_per_order + self.o_time +
#                             [self.received_order_x, self.received_order_y] + [self.received_order_reward] +
#                              [self.dr_left_capacity] + [self.clock]

#dvrp_v_8 acceptance order separately,no info about new order inside orders, except  o_statuses, PPO-11
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
env.seed(1)
set_random_seed(1)
t1 = torch.get_rng_state()
t2 = numpy.random.get_state()
env = ActionMasker(env, mask_fn)  # Wrap to enable masking

path = "./a2c_cartpole_tensorboard/"
model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1, tensorboard_log=path)
model.learn(total_timesteps=60000000, log_interval=100) #6 deleted
model.save("dvrp_v_8")


# #
# torch.set_rng_state(t1)
# numpy.random.set_state(t2)
# model = MaskablePPO.load("dvrp_v_2", env = env)
# obs = env.reset()
#
# total_reward = 0
# for i in range(700):
#     action_masks = mask_fn(env)
#     action, _states = model.predict(obs, action_masks=action_masks)
#     # if (action == 0):
#     #     print("QQQQQ")
#     obs, rewards, dones, info = env.step(action)
#     total_reward += rewards
#     env.render()
# print('total_reward', total_reward)
# print('__________________')
# #
# obs = env.reset()
# for i in range(700):
#     action_masks = mask_fn(env)
#     action, _states = model.predict(obs, action_masks=action_masks)
#     # if (action == 0):
#     #     print("QQQQQ")
#     obs, rewards, dones, info = env.step(action)
#     env.render()
# print('__________________')
# # #
# obs = env.reset()
# for i in range(700):
#     action_masks = mask_fn(env)
#     action, _states = model.predict(obs, action_masks=action_masks)
#     # if (action == 0):
#     #     print("QQQQQ")
#     obs, rewards, dones, info = env.step(action)
#     env.render()
# print('__________________')
# #
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
















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

