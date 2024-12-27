import numpy as np
import gymnasium as gym
import random
from load_data import load_training_data, write_training_data

env = gym.make('FrozenLake-v1', render_mode='ansi', is_slippery=False)
loaded_data = None
num_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.5
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01
rewards_all_episodes = []

def q_learning(rounds, exploration_rate, q_table):
    # Q-learning algorithm
    for episode in range(rounds):
        # initialize new episode params
        state = env.reset()[0]
        done = False
        rewards_current_episode = 0

        for step in range(max_steps_per_episode): 
            # Exploration-exploitation trade-off
            # Take new action
            # Update Q-table
            # Set new state
            # Add new reward        
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
                break
        
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)    
        rewards_all_episodes.append(rewards_current_episode)
    return q_table, rewards_all_episodes, exploration_rate

def main():

    file_path = "training_data/q_table_v1.npz"

    loaded_data = load_training_data(file_path, env)
    q_table = loaded_data["q_table"]
    print("\n\n********Old Q-table********\n")
    print(q_table)
    exploration_rate = loaded_data["exploration_rate"]
    
    q_table, rewards_all_episodes, exploration_rate = q_learning(num_episodes, exploration_rate, q_table)
    rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
    print(exploration_rate)
    count = 1000
    print("********Average reward per thousand episodes********\n")
    for r in rewards_per_thousand_episodes:
        print(count, ": ", str(sum(r/1000)))
        count += 1000

    print("\n\n********Q-table********\n")
    print(q_table)
    
    
    write_training_data(file_path, q_table, exploration_rate)


# Using the special variable 
# __name__
if __name__=="__main__":
    main()