"""
Reinforcement learning maze example.

This script is the main part which controls the update method of this example using q-learning algorithm.
The RL algorithm (Q-learning) is in RL_agent.py.
The environment is presented in maze_env.py.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

"""

from maze_env import Maze
from RL_agent import QLearningTable
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

episode_count = 50 # Number of episodes to run the experiment
episodes = range(episode_count)
movements = [] # Number of movements happened in each episode
rewards = [] # The gained reward in each episode

'''
This function updates the position of the explorer in the Maze environment based on the actions it chooses.
'''
def run_experiment():

    for episode in episodes:
        print("Episode %s/%s." %(episode+1, episode_count))
        # initial observation;
        observation = env.reset()
        moves = 0

        while True:
            # fresh env
            env.render()

            # Q-learning chooses action based on observation
            # we convert observation to str since we want to use them as index for our DataFrame.
            action = q_learning_agent.choose_action(str(observation)) # ToDo: call choose_action() method from the agent QLearningTable instance

            # RL takes action and gets next observation and reward
            observation_, reward, done = env.get_state_reward(action) # ToDo: call get_state_reward() method from Maze environment instance
            moves +=1

            # RL learn from the above transition,
            # Update the Q value for the given tuple
            q_learning_agent.learn(str(observation), action, reward, str(observation_))# ToDo: call learn method from Q-learning agent instance, passing (s, a, r, s') tuple

            # consider the next observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                movements.append(moves) # Keep track of the number of movements in this episode
                rewards.append(reward) # Keep track of the gained reward in this episode
                print("Reward: {0}, Moves: {1}".format(reward, moves))
                break

    # end of game
    print('game over!')
    # Show the results
    plot_reward_movements()

def plot_reward_movements():
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(episodes, movements)
    plt.xlabel("Episode")
    plt.ylabel("# Movements")

    plt.subplot(2,1,2)
    plt.step(episodes, rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("rewards_movements_q_learn.png")
    plt.show()


if __name__ == "__main__":

    # Craete maze environment
    env = Maze() #ToDo: instanciate Maze class

    # Create Q-learning agent
    q_learning_agent = QLearningTable(actions=list(range(env.n_actions))) #ToDo: instanciate QLearningTable class

    # Call run_experiment() function once after given time in milliseconds.
    env.window.after(10, run_experiment)

    # The infinite loop used to run the application, wait for an event to occur and process the event
    # till the window is not closed.
    env.window.mainloop()
