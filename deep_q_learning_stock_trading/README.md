# Q-Trader

An implementation of Q-learning applied to (short-term) stock trading. The model uses n-day windows of closing prices to determine if the best action to take at a given time is to buy, sell or sit.

As a result of the short-term state representation, the model is not very good at making decisions over long-term trends, but is quite good at predicting peaks and troughs.


### Results
 We trained the model woth GSPC data of 2010 and tested with the first quarter of 2011.

S&P 500, 2011Q1. Profit of $92.84:
 
![^GSPC 2010](./images/buy_sell.png)


## Running the Code

```
mkdir models
python train_app.py
```
You may change the these parameters in train_app.py:
<pre>window_size = 5
episode_count = 30
stock_name = "^GSPC_2011"
</pre>

Then when training finishes you can evaluate with the test dataset :
```
python evaluate_app.py
```
Change these variables in evaluate_app.py accordingly before running:
<pre>
stock_name = "GSPC_2011-03"
model_name = "model_ep30"
</pre>
