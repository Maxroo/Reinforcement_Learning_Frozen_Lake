#!/bin/env python
import numpy as np
import gymnasium as gym
import random
import time
from IPython.display import clear_output
import sys
from load_data import load_training_data, write_training_data

from gymnasium.wrappers import RecordVideo
from gym_to_gif import save_frames_as_gif


env = gym.make('FrozenLake-v1', render_mode='rgb_array', is_slippery=False)
env = RecordVideo(env, './video/', video_length=0)
env.start_recording("pre_training_video")
num_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001
rewards_all_episodes = []

file_path = "training_data/q_table_v1.npz"


loaded_data = load_training_data(file_path, env)
q_table = loaded_data["q_table"]
exploration_rate = loaded_data["exploration_rate"]
print(exploration_rate)

round = 1
if len(sys.argv) > 1 :
    round = int(sys.argv([1]))

frames = []
# Q-learning algorithm
for episode in range(round):
    # initialize new episode params
    state = env.reset()[0]
    done = False
    rewards_current_episode = 0
    print(f'*********Round {episode+1}*********')
    for step in range(max_steps_per_episode): 
        # Exploration-exploitation trade-off
        # Take new action
        # Update Q-table
        # Set new state
        # Add new reward
        clear_output(wait=True)
        frame = env.render()
        frames.append(frame)
        time.sleep(0.3)
        
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:]) 
        else:
            action = env.action_space.sample()
        # Exploration rate decay   
        # Add current episode reward to total rewards list

        new_state, reward, done, truncated, info = env.step(action)

        # Update Q-table for Q(s,a)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
        learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        state = new_state
        rewards_current_episode += reward 
        if done == True: 
            clear_output(wait=True)
            frame = env.render()
            frames.append(frame)
            if reward: 
                print("*********You reached the goal!*********")
                time.sleep(3)
            else:
                print("*********You lost!*********")
                time.sleep(3)
            break
    rewards_all_episodes.append(rewards_current_episode)

write_training_data(file_path, q_table, exploration_rate)
print("\n\n********Q-table********\n")
print(q_table)
env.close()

save_frames_as_gif(frames, filename="after.gif")