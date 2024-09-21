# import pandas as pd
# import numpy as np
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from sklearn.metrics import mean_squared_error
# from math import sqrt
# import matplotlib.pyplot as plt

# # Load the data
# df = pd.read_csv('hackathon_sample_v2.csv')

# # Convert 'date' column to datetime format
# df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

# # Use 'stock_exret' as the returns column
# returns_column = 'stock_exret'

# # Resample to monthly frequency and calculate mean returns
# df_monthly = df.set_index('date')[returns_column].resample('M').mean()

# # Split the data into train and test sets
# train_end_date = '2023-06-30'
# train = df_monthly[:train_end_date]
# test_start_date = '2021-07-01'
# test = df_monthly[test_start_date:]

# # Fit the SARIMA model
# model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
# results = model.fit()

# # Make predictions for the test period using explicit dates
# predictions = results.get_prediction(start=test.index[0], end=test.index[-1]).predicted_mean

# # Calculate RMSE
# rmse = sqrt(mean_squared_error(test, predictions))
# print(f'RMSE: {rmse}')

# # Plot the results
# plt.figure(figsize=(12, 6))
# plt.plot(train.index, train, label='Train')
# plt.plot(test.index, test, label='Test')
# plt.plot(test.index, predictions, label='Predictions')
# plt.legend()
# plt.title('SARIMA Model: Actual vs Predicted')
# plt.show()

# # Stock selection based on latest data
# latest_date = df['date'].max()
# latest_data = df[df['date'] == latest_date]

# # Rank stocks by expected return
# latest_data['rank'] = latest_data[returns_column].rank(ascending=False)

# # Select top 50 stocks for long positions and bottom 50 for short positions
# top_stocks = latest_data.nsmallest(50, 'rank')
# bottom_stocks = latest_data.nlargest(50, 'rank')

# # Display selected stocks
# print("Top Stocks to Buy:")
# print(top_stocks[['stock_ticker', returns_column]])

# print("\nBottom Stocks to Short:")
# print(bottom_stocks[['stock_ticker', returns_column]])


# Code to analyze Individual Stock (MSFT in this case)

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('hackathon_sample_v2.csv')

# Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

# Filter data for MSFT
msft_data = df[df['stock_ticker'] == 'MSFT']

# Use 'stock_exret' as the returns column
returns_column = 'stock_exret'

# Resample to monthly frequency and calculate mean returns
msft_monthly = msft_data.set_index('date')[returns_column].resample('M').mean()

# Split the data into train and test sets
train_end_date = '2023-06-30'
train = msft_monthly[:train_end_date]
test_start_date = '2021-07-01'
test = msft_monthly[test_start_date:]

# Fit the SARIMA model
model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Make predictions for the test period using explicit dates
predictions = results.get_prediction(start=test.index[0], end=test.index[-1]).predicted_mean

# Calculate RMSE
rmse = sqrt(mean_squared_error(test, predictions))
print(f'RMSE: {rmse}')

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(test.index, predictions, label='Predictions')
plt.legend()
plt.title('SARIMA Model: Actual vs Predicted for MSFT')
plt.show()
