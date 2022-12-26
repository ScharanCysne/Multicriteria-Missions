import json 
import pandas
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.results_plotter import X_EPISODES, ts2xy, window_func

def save_rewards(log_dir, num_drones, num_episodes):
    file_name = f"{num_drones}_{num_episodes}.monitor.csv"
    headers = []
    with open(log_dir + file_name, "rt") as file_handler:
        first_line = file_handler.readline()
        assert first_line[0] == "#"
        header = json.loads(first_line[1:])
        data_frame = pandas.read_csv(file_handler, index_col=None)
        headers.append(header)
        data_frame["t"] += header["t_start"]
    data_frame.sort_values("t", inplace=True)
    data_frame["t"] -= min(header["t_start"] for header in headers)

    (x,y) = ts2xy(data_frame, X_EPISODES)

    plt.figure("Coverage Mission", figsize=(8, 2))
    max_x = x[-1]
    min_x = 0
    plt.scatter(x, y, s=2)
    # Compute and plot rolling mean with window of size EPISODE_WINDOW
    x, y_mean = window_func(x, y, 25, np.mean)
    plt.plot(x, y_mean)
    plt.xlim(min_x, max_x)
    plt.title(f"Coverage Mission | {num_drones} Drones | {num_episodes} Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.tight_layout()
    plt.savefig(log_dir + f"rewards_{num_drones}_{num_episodes}")