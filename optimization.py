import pandas as pd
import numpy as np
import yfinance as yf

# ============================
# 1. Load and Merge Data
# ============================

# Load the provided CSV files
elastic_net_coefficients = pd.read_csv('/Users/Ray.Fang/Downloads/Elastic Net Regression (Fama French)/elastic_net_coefficients_before2010.csv')
famafrench_factors = pd.read_csv('/Users/Ray.Fang/Downloads/Elastic Net Regression (Fama French)/famafrench.csv')
testdata = pd.read_csv('/Users/Ray.Fang/Downloads/Elastic Net Regression (Fama French)/testdata.csv')

# Merge the test data and elastic net coefficients on Stock Ticker
merged_data = pd.merge(testdata, elastic_net_coefficients, left_on="stock_ticker", right_on="Stock Ticker")

# Calculate the expected return for each stock based on the Fama French model
latest_factors = famafrench_factors.iloc[-1]  # Assuming the last row has the latest factors
merged_data['expected_return'] = (
    merged_data['Beta_1 (SMB)'] * latest_factors['SMB'] +
    merged_data['Beta_2 (HML)'] * latest_factors['HML'] +
    merged_data['Beta_3 (RMW)'] * latest_factors['RMW'] +
    merged_data['Beta_4 (CMA)'] * latest_factors['CMA'] +
    merged_data['Beta_5 (RF)'] * latest_factors['RF']
)

# ============================
# 2. Prepare Covariance Matrix
# ============================

# Convert the excess returns in the test data to numeric for covariance calculation
testdata['stock_exret'] = pd.to_numeric(testdata['stock_exret'], errors='coerce')

# Merge test data and elastic net coefficients to align data
merged_data_full = pd.merge(testdata, elastic_net_coefficients, left_on="stock_ticker", right_on="Stock Ticker")

# Filter out stocks without valid excess returns
filtered_data = merged_data_full.dropna(subset=['stock_exret'])

# Create a pivot table of excess returns for the covariance matrix (stocks vs. dates)
returns_pivot = filtered_data.pivot_table(index='date', columns='stock_ticker', values='stock_exret')

# Clean the covariance matrix by dropping stocks with any missing data
returns_pivot_clean = returns_pivot.dropna(axis=1, how='any')
cov_matrix_clean = returns_pivot_clean.cov()

# ============================
# 3. Select and Weight Stocks
# ============================

# Ensure that the stocks we select have clean data
available_tickers = cov_matrix_clean.columns.tolist()

# Filter merged_data to include only available tickers and drop duplicates
available_merged_data = merged_data[merged_data['stock_ticker'].isin(available_tickers)].drop_duplicates('stock_ticker')

# Desired number of stocks to select
desired_num_stocks = 100

# Check if we have enough stocks to select
if len(available_merged_data) < desired_num_stocks:
    print(f"Not enough stocks with complete data to select {desired_num_stocks} stocks.")
    selected_num_stocks = len(available_merged_data)
    print(f"Selecting {selected_num_stocks} stocks instead.")
else:
    selected_num_stocks = desired_num_stocks

# Randomly select stocks from the available ones
selected_stocks = available_merged_data.sample(n=selected_num_stocks, random_state=42)  # random_state for reproducibility

# Assign random weights and normalize to sum to 1 for calculations
random_weights = np.random.random(selected_num_stocks)
random_weights /= random_weights.sum()  # Normalize to sum to 1

# Add the weights to the selected stocks dataframe
selected_stocks = selected_stocks.copy()  # Avoid SettingWithCopyWarning
selected_stocks['weight_decimal'] = random_weights  # Decimal form for calculations
selected_stocks['weight_percent'] = random_weights * 100  # Percentage form for CSV output

# ============================
# 4. Calculate Portfolio Metrics
# ============================

# Recalculate the expected portfolio return using the selected stocks
portfolio_expected_return = np.sum(selected_stocks['expected_return'] * selected_stocks['weight_decimal'])

# Ensure that the covariance matrix aligns with the selected stocks
selected_tickers = selected_stocks['stock_ticker'].tolist()
selected_cov_matrix = cov_matrix_clean.loc[selected_tickers, selected_tickers]

# Calculate portfolio variance and standard deviation
weights_array = selected_stocks['weight_decimal'].values
portfolio_variance = np.dot(weights_array.T, np.dot(selected_cov_matrix.values, weights_array))
portfolio_std_dev = np.sqrt(portfolio_variance)

# Latest risk-free rate from the Fama French factors
latest_risk_free_rate = latest_factors['RF']

# Sharpe ratio calculation
sharpe_ratio = (portfolio_expected_return - latest_risk_free_rate) / portfolio_std_dev

# ============================
# 5. Output Results
# ============================

# Output results
print(f"Portfolio Expected Return: {portfolio_expected_return:.4f}")
print(f"Portfolio Risk (Std Dev): {portfolio_std_dev:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")

# ============================
# 6. Save to CSV
# ============================

# Save selected stocks and weights to CSV (weights in percentages)
selected_stocks[['stock_ticker', 'expected_return', 'weight_percent']].to_csv('updated_stock_selection_with_weights.csv', index=False)
selected_stocks[['stock_ticker', 'weight_percent']].to_csv('stock_ticker_weights.csv', index=False)

# Verify that weights sum to 100%
total_weight_percent = selected_stocks['weight_percent'].sum()
print(f"Total Weight (should be 100%): {total_weight_percent:.2f}%")
