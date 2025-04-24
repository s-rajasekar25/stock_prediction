import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from datetime import datetime

df = pd.read_csv("C:/Users/sraja/Downloads/stock_prices.csv")


df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

result = adfuller(df['Close'])
print(f"ADF Test Statistic: {result[0]}")
print(f"P-Value: {result[1]}")

plt.figure(figsize=(10,5))
plt.plot(df['Close'])
plt.title("Stock Price Over Time")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()

model = ARIMA(df['Close'], order=(5,1,0)) 
model_fit = model.fit()

forecast_steps = 30
forecast = model_fit.forecast(steps=forecast_steps)

plt.figure(figsize=(10,5))
plt.plot(df.index[-100:], df['Close'].iloc[-100:], label="Actual Prices")
plt.plot(pd.date_range(df.index[-1], periods=forecast_steps, freq='D'), forecast, label="Forecasted Prices", linestyle="dashed")
plt.title("Stock Price Prediction using ARIMA")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.show()