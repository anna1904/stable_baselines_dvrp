from packaging import version
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb
# from scipy.ndimage.filters import gaussian_filter1d
import numpy as np

##1.0 How to read events from the event_file:
# from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
# for event in EventFileLoader('sprint_1/PPO_7/events.out.tfevents.1679831776.network154-009.hiMolde.no.18388.0').Load():
#     print(event)

##2.0 Create experiment to tensorboard_dev (command  tensorboard dev upload --logdir ./sprint_1/)
# experiment_id = "A6X2kyb2TGqZ9xkM4wD06g"
# experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
# df = experiment.get_scalars()
#
# dfw = experiment.get_scalars()
csv_path = 'csv_files/PPO_7.csv'
# dfw.to_csv(csv_path, index=False)

matplotlib.use('TkAgg')
df = pd.read_csv(csv_path)
filtered_df = df.loc[(df['run'] == 'PPO_7') & (df['tag'].str.contains('rollout/ep_rew_mean'))]

## 3.0 Do this gaussing curve, couldnt install scipy.ndimage
# err = (1 - filtered_df['value']) / 2
# filtered_df['value'] += np.random.normal(0, err / 10, filtered_df['value'].size)
#
# upper = gaussian_filter1d(filtered_df['value'] + err, sigma=3)
# lower = gaussian_filter1d(filtered_df['value'] - err, sigma=3)
#
# fig, ax = plt.subplots(ncols=2)
#
# ax[0].errorbar(filtered_df['step'], filtered_df['value'], err, color='dodgerblue')
#
# ax[1].plot(filtered_df['step'], filtered_df['value'], color='dodgerblue')
# ax[1].fill_between(filtered_df['step'], upper, lower, color='crimson', alpha=0.2)
#
# plt.show()

##4.0 How to filter outliers:
# Q1 = filtered_df['value'].quantile(0.25)
# Q3 = filtered_df['value'].quantile(0.75)
# print(Q1, Q3)
# IQR = Q3 - Q1
# filtered_df = filtered_df[(filtered_df['value'] > (Q1 - 1.5 * IQR)) & (filtered_df['value'] < (Q3 + 1.5 * IQR))]
#
sns.lineplot(x='step', y='value', data=filtered_df)

# Set the labels and title for the plot
plt.xlabel('Step')
plt.ylabel('Value')
plt.title('PPO_7 - rollout/ep_len_mean')
plt.show()


## 5.0 CONTINUE TRAINING!!!
# from stable_baselines3 import A2C
#
# model = A2C("MlpPolicy", "CartPole-v1", verbose=1, tensorboard_log="./a2c_cartpole_tensorboard/")
# model.learn(total_timesteps=10_000, tb_log_name="first_run")
# # Pass reset_num_timesteps=False to continue the training curve in tensorboard
# # By default, it will create a new curve
# # Keep tb_log_name constant to have continuous curve (see note below)
# model.learn(total_timesteps=10_000, tb_log_name="second_run", reset_num_timesteps=False)
# model.learn(total_timesteps=10_000, tb_log_name="third_run", reset_num_timesteps=False)

