import numpy as np
import os

def write_training_data(file_path, q_table, exploration_rate):
    np.savez(file_path, exploration_rate = exploration_rate , q_table = q_table)

def load_training_data(file_path, env):
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n
    if not os.path.exists(file_path): 
        return {"q_table": np.zeros((state_space_size, action_space_size)), "exploration_rate" : 1}
    return np.load(file_path) # load