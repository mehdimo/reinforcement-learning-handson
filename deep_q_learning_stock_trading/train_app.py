from agent import Agent
from market_env import Market

import os
import time

def main():

    window_size = 5
    episode_count = 10
    stock_name = "^GSPC_2011"

    agent = Agent(window_size)
    market = Market(window_size=window_size, stock_name=stock_name)

    batch_size = 32

    start_time = time.time()
    for e in range(episode_count + 1):
        print("Episode " + str(e) + "/" + str(episode_count))
        agent.reset()
        state, price_data = market.reset() #ToDo: get the initial state

        for t in range(market.last_data_index):
            # get the action of the agent
            action, bought_price = agent.act(state, price_data) # ToDo: Call the act() method of the agent considering the current state

            # get the next state of the stock
            #ToDo: Get the next available state from market data
            next_state, next_price_data, reward, done = market.get_next_state_reward(action, bought_price)

            #ToDo: add the transaction to the memory
            agent.memory.append((state, action, reward, next_state, done))
            # learn from the history
            if len(agent.memory) > batch_size:
                agent.experience_replay(batch_size)

            state = next_state
            price_data = next_price_data

            if done:
                print("--------------------------------")
                print("Total Profit: {0}".format(agent.get_total_profit()))
                print("--------------------------------")

        if e % 10 == 0:
            if not os.path.exists("models"):
                os.mkdir("models")
            agent.model.save("models/model_ep" + str(e))

    end_time = time.time()
    training_time = round(end_time - start_time)
    print("Training took {0} seconds.".format(training_time))

if __name__ == "__main__":
    main()
