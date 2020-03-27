from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque

class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.__inventory = []
        self.__total_profit = 0
        self.action_history = []

        self.state_size = state_size  # normalized previous days
        self.action_size = 3  # sit, buy, sell
        self.memory = deque(maxlen=1000)
        self.model_name = model_name
        self.is_eval = is_eval

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.model = load_model("models/" + model_name) if is_eval else self.create_model()

    # Create deep learning model structure
    def create_model(self):
        model = Sequential()
        model.add(Dense(units=32, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001))

        return model

    def reset(self):
        self.__inventory = []
        self.__total_profit = 0
        self.action_history = []

    def act(self, state, price_data):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            #ToDo: select a random action
            action = random.randrange(self.action_size)
        else:
            # Predict what would be the possible action for a given state
            options = self.model.predict(state) #ToDo: predict q-value of the current state

            # pick the action with highest probability
            action = np.argmax(options[0]) #ToDo: select the q-value with highest value

        bought_price = None
        if action == 0:  # Do nothing!
            print(".", end='', flush=True)
            self.action_history.append(action)
        elif action == 1:  # buy
            self.buy(price_data)
            self.action_history.append(action)
        elif action == 2 and self.has_inventory():  # sell
            bought_price = self.sell(price_data)
            self.action_history.append(action)
        else: # action is 2 (sell) but we don't have anything in inventory to sell!
            self.action_history.append(0)

        return action, bought_price

    def experience_replay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])

        for state, action, reward, next_state, done in mini_batch:
            if done:
                target = reward
            else:
                # updated q_value = reward + gamma * [max_a' Q(s',a')]
                next_q_values = self.model.predict(next_state)[0] # this is Q(s', a') for all possible a'
                target = reward + self.gamma * np.amax(next_q_values) # ToDo: update target q_value using Bellman equation

            predicted_target = self.model.predict(state) #ToDo: predict q_value for current state
            # Update the action values
            predicted_target[0][action] = target # ToDo: Substitue target q_value to the predicted value
            # Train the model with updated action values
            self.model.fit(state, predicted_target, epochs=1, verbose=0) #ToDo: train the model with new q_value

        # Make epsilon smaller over time, so do more exploitation than exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def buy(self, price_data):
        self.__inventory.append(price_data)
        print("Buy: {0}".format(self.format_price(price_data)))

    def sell(self, price_data):
        bought_price = self.__inventory.pop(0)
        self.__total_profit += price_data - bought_price
        print("Sell: {0} | Profit: {1}".format(self.format_price(price_data), self.format_price(price_data - bought_price)))
        return bought_price

    def get_total_profit(self):
        return self.format_price(self.__total_profit)

    def has_inventory(self):
        return len(self.__inventory) > 0

    def format_price(self, n):
        return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))
