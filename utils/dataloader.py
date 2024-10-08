import pandas as pd
import yfinance as yf

# Fetch data for a specific stock
ticker = input("Enter symbol: ")
data = yf.download(ticker, start="2024-01-01", end="2024-07-01")
print(data.head())

# Save the data to a CSV file
data.to_csv(f'data/{ticker}_historical_data.csv')
