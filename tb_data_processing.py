
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
def smooth_data(data, window_size):
    padding_size = window_size // 2
    padded_data = np.pad(data, (padding_size, padding_size), mode='edge')
    window = np.ones(window_size) / window_size
    smoothed_data = np.convolve(padded_data, window, mode='same')[padding_size:-padding_size]
    return smoothed_data

csv_path = ['csv_files/PPO_7.csv', 'csv_files/PPO_8.csv', 'csv_files/PPO_9.csv', 'csv_files/PPO_10.csv', 'csv_files/PPO_11.csv', 'csv_files/PPO_12.csv']


matplotlib.use('TkAgg')

def read_and_plot(window_size, csv_path):
    plt.figure(figsize=(10, 6))
    rs = 0
    for i in csv_path:
        rs += 1
        df = pd.read_csv(i)
        smoothed_values = smooth_data(df['Value'], window_size)
        sns.lineplot(x=df['Step'], y=smoothed_values, label=f'{rs}')

        # Set labels and title
        plt.xlim(120, df['Step'].max())
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.title('Effect of Different Random Seeds for Basic Observation Space')
    plt.legend()
    plt.show()

read_and_plot(100, csv_path)


