import pandas as pd
import numpy as np
import yfinance as yf

# Fetch the list of S&P 500 companies
try:
    # Use yfinance to get S&P 500 tickers
    sp500_tickers = yf.Tickers(' '.join(yf.Ticker('^GSPC').symbols)).symbols
except Exception:
    # Alternative method if the above fails
    import requests
    tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    sp500_table = tables[0]
    sp500_tickers = sp500_table['Symbol'].tolist()

# Ensure we have the tickers
if not sp500_tickers or len(sp500_tickers) < 100:
    print("Error fetching S&P 500 tickers. Please check your internet connection and try again.")
    exit()

# Randomly select 100 tickers
np.random.seed(0)  # For reproducibility
selected_tickers = np.random.choice(sp500_tickers, 100, replace=False)

# Generate random weights that sum to 100%
weights = np.random.rand(100)
weights = weights / weights.sum() * 100  # Scale weights to sum to 100%

# Create DataFrame
portfolio_df = pd.DataFrame({
    'Tickers': selected_tickers,
    'Weights': weights
})

# Ensure weights sum to 100%
total_weight = portfolio_df['Weights'].sum()
print(f"Total Weight: {total_weight:.2f}%")

# Save to CSV
portfolio_df.to_csv('Portfolio.csv', index=False)
print("Portfolio.csv has been created with 100 tickers and random weights.")
