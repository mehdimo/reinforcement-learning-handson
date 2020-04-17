import numpy as np
import pandas as pd
from sklearn import preprocessing

class Market:
    def __init__(self, window_size, stock_name):
        self.data = self.__get_stock_data(stock_name)
        self.states = self.__get_all_window_prices_diff(self.data, window_size)
        self.index = -1
        self.last_data_index = len(self.data) - 1

    def __get_stock_data(self, key):
        file_path = "data/" + key + ".csv"
        lines = pd.read_csv(file_path, sep=',')
        print(lines.head())
        print("all data loaded.")
        return lines

    def __get_window(self, data_df, t, n):
        d = t - n + 1
        data1 = data_df["Close"].values
        data2 = data_df["Volume"].values
        block1 = data1[d:t + 1] if d >= 0 else np.append(-d * [data1[0]], data1[0:t + 1])  # pad with t0
        block2 = data2[d:t + 1] if d >= 0 else np.append(-d * [data2[0]], data2[0:t + 1])
        res = []
        for i in range(n - 1):
            res.append(block1[i + 1] - block1[i])
        for i in range(n - 1):
            res.append(block2[i + 1] - block2[i])
        return np.array([res])

    def normalize_data(self, in_df):
        x = in_df.values  # get the numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled, columns=in_df.columns)
        return df

    # Preprocess data to create a list of window-size states
    def __get_all_window_prices_diff(self, data, n):
        l = len(data)
        processed_data = []

        sel_col = ["Close", "Volume"]
        scaled_data = self.normalize_data(data[sel_col])

        for t in range(l):
            state = self.__get_window(scaled_data, t, n + 1)
            processed_data.append(state)
        return processed_data

    def reset(self):
        self.index = -1
        return self.states[0], self.data.iloc[0]["Close"]

    def get_next_state_reward(self, action, bought_price=None):
        self.index += 1
        if self.index > self.last_data_index:
            self.index = 0
        next_state = self.states[self.index + 1]
        next_price_data = self.data.iloc[self.index + 1]["Close"]
        price_data = self.data.iloc[self.index]["Close"]
        reward = 0
        if action==2 and bought_price is not None:
            reward = max(price_data - bought_price, 0)

        done = True if self.index == self.last_data_index - 1 else False

        return next_state, next_price_data, reward, done
