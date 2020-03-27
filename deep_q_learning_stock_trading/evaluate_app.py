from keras.models import load_model

from agent import Agent
from market_env import Market

import matplotlib

#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def main():

    stock_name = "GSPC_2011-03"
    model_name = "model_ep10"

    model = load_model("models/" + model_name)
    window_size = model.layers[0].input.shape.as_list()[1]

    agent = Agent(window_size, True, model_name)
    market = Market(window_size, stock_name)

    state, price_data = market.reset() #ToDo: Start from an initial state

    for t in range(market.last_data_index):
        action, bought_price = agent.act(state, price_data) # ToDo: Get action for the current state

        # Check the action to get reward and observe next state
        next_state, next_price_data, reward, done = market.get_next_state_reward(action, bought_price) #ToDo: get next state

        state = next_state
        price_data = next_price_data

        if done:
            print("--------------------------------")
            print("{0} Total Profit: {1}".format(stock_name, agent.get_total_profit()))
            print("--------------------------------")

    plot_action_profit(market.data, agent.action_history, agent.get_total_profit())

def plot_action_profit(data, action_data, profit):
    plt.plot(range(len(data)), data)
    plt.xlabel("date")
    plt.ylabel("price")
    buy, sel = False, False
    for d in range(len(data) -1):
        if action_data[d] == 1:  # buy
            buy, = plt.plot(d, data[d], 'g*')
        elif action_data[d] == 2:  # sell
            sel, = plt.plot(d, data[d], 'r+')
    if buy and sel:
        plt.legend([buy, sel], ["Buy", "Sell"])
    plt.title("Total Profit: {0}".format(profit))
    plt.savefig("buy_sell.png")
    plt.show()

if __name__=="__main__":
    main()