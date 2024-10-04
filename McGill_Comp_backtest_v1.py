import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from matplotlib.ticker import PercentFormatter

# ============================
# 1. Load and Prepare Weightings
# ============================

# Read the CSV file
df = pd.read_csv('stock_ticker_weights.csv')

# Drop duplicate tickers, keeping only the first occurrence
df = df.drop_duplicates(subset='Tickers', keep='first')

# Create the weightings dictionary
weightings1 = dict(zip(df['Tickers'], df['Weights']))
weightings2 = {"SPY": 100}  # Benchmark

# Initial list of tickers to fetch data for
members = list(weightings1.keys()) + ["SPY"]

# ============================
# 2. Fetch Historical Price Data
# ============================

def fetch_price_data(ticker):
    """
    Fetch historical adjusted close prices for a given ticker.
    Returns a DataFrame with 'Date' and 'Close' columns.
    """
    try:
        ticker_data = yf.Ticker(ticker).history(period="max").reset_index()[["Date", "Close"]]
        if ticker_data.empty:
            print(f"No data found for ticker: {ticker}")
            return None
        ticker_data["Date"] = pd.to_datetime(ticker_data["Date"])
        ticker_data = ticker_data.rename(columns={"Close": ticker})
        return ticker_data
    except Exception as e:
        print(f"Error fetching data for ticker {ticker}: {e}")
        return None

# Initialize basedata with the first ticker
initial_ticker = members[0]
print(f"Fetching data for {initial_ticker}")
basedata = fetch_price_data(initial_ticker)

if basedata is None:
    print(f"Error: No data fetched for the initial ticker {initial_ticker}. Exiting.")
    exit()

# Fetch and merge data for the remaining tickers
for ticker in members[1:]:
    print(f"Fetching data for {ticker}")
    ticker_data = fetch_price_data(ticker)
    if ticker_data is None:
        print(f"Removing {ticker} from weightings and members due to missing data.")
        if ticker in weightings1:
            del weightings1[ticker]
        continue
    basedata = pd.merge(basedata, ticker_data, on="Date", how='inner')

# Update members list after removing tickers without data
members = list(weightings1.keys()) + ["SPY"]

# ============================
# 3. Filter Data by Date
# ============================

# Check if basedata is empty after merging
print("basedata shape:", basedata.shape)
if basedata.empty:
    print("Error: basedata is empty after merging data.")
    exit()

# Define start date for analysis
start_date = "2010-01-01"
basedata = basedata[basedata["Date"] > start_date]
basedata.reset_index(drop=True, inplace=True)

print(f"Data date range after filtering (>{start_date}):")
print("Start date:", basedata["Date"].min())
print("End date:", basedata["Date"].max())

if basedata.empty:
    print("Error: basedata is empty after date filtering.")
    exit()

# ============================
# 4. Normalize Price Data
# ============================

# Function to normalize ticker prices
def normalize_prices(df, tickers):
    """
    Normalize each ticker's price by dividing by its first available price.
    """
    valid_tickers = []
    for ticker in tickers:
        if ticker not in df.columns:
            print(f"Ticker {ticker} not found in basedata columns. Skipping normalization.")
            continue
        if df[ticker].empty:
            print(f"Ticker {ticker} has no data. Skipping normalization.")
            continue
        try:
            df[ticker] = df[ticker] / df[ticker].iloc[0]
            valid_tickers.append(ticker)
        except IndexError:
            print(f"IndexError: Ticker {ticker} has no data to normalize. Skipping.")
    return df, valid_tickers

# Normalize prices and get list of tickers with successful normalization
basedata, valid_tickers = normalize_prices(basedata, members)

# Update weightings1 to include only valid tickers
weightings1 = {ticker: weightings1[ticker] for ticker in weightings1 if ticker in valid_tickers}

# Re-normalize weights to sum to 100%
total_weight = sum(weightings1.values())
if total_weight != 100:
    weightings1 = {ticker: (weight / total_weight) * 100 for ticker, weight in weightings1.items()}
    print("Weights have been rebalanced to sum to 100%.")

# Update members list after normalization
members = list(weightings1.keys()) + ["SPY"]

# ============================
# 5. Define Backtester Function
# ============================

def Backtester(weightings, data, name):
    """
    Calculate portfolio value based on provided weights.
    """
    portfolio_value = 0
    for ticker, weight in weightings.items():
        if ticker in data.columns:
            portfolio_value += float(weight) * data[ticker] / 100
        else:
            print(f"Warning: Ticker {ticker} not found in data. Skipping in portfolio calculation.")
    data[name] = portfolio_value
    return data

# ============================
# 6. Calculate Portfolio Values
# ============================

# Run Backtester for Portfolio1 (Your Portfolio)
basedata = Backtester(weightings1, basedata, "Portfolio1")

# Run Backtester for Portfolio2 (Benchmark: SPY)
basedata = Backtester(weightings2, basedata, "Portfolio2")

# ============================
# 7. Calculate Returns
# ============================

basedata['Portfolio1_Returns'] = basedata['Portfolio1'].pct_change()
basedata['Portfolio2_Returns'] = basedata['Portfolio2'].pct_change()
basedata = basedata.dropna(subset=['Portfolio1_Returns', 'Portfolio2_Returns'])

# ============================
# 8. Define Performance Metrics Functions
# ============================

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    mean_return = returns.mean()
    std_return = returns.std()
    annualized_return = mean_return * 252  # 252 trading days
    annualized_volatility = std_return * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    return annualized_return, annualized_volatility, sharpe_ratio

def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    mean_return = returns.mean()
    downside_std = returns[returns < 0].std()
    annualized_return = mean_return * 252  # 252 trading days
    annualized_downside_volatility = downside_std * np.sqrt(252)
    sortino_ratio = (annualized_return - risk_free_rate) / annualized_downside_volatility
    return sortino_ratio

def calculate_alpha_beta(portfolio_returns, benchmark_returns, risk_free_rate=0.02):
    daily_risk_free_rate = risk_free_rate / 252
    portfolio_excess_returns = portfolio_returns - daily_risk_free_rate
    benchmark_excess_returns = benchmark_returns - daily_risk_free_rate
    covariance_matrix = np.cov(portfolio_excess_returns, benchmark_excess_returns)
    beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
    alpha = portfolio_excess_returns.mean() - beta * benchmark_excess_returns.mean()
    annualized_alpha = alpha * 252
    return annualized_alpha, beta

def calculate_information_ratio(portfolio_returns, benchmark_returns):
    excess_returns = portfolio_returns - benchmark_returns 
    mean_excess_return = excess_returns.mean() * 252
    tracking_error = excess_returns.std() * np.sqrt(252)
    information_ratio = mean_excess_return / tracking_error
    return information_ratio, excess_returns

def calculate_max_drawdown(portfolio_values):
    cumulative_max = portfolio_values.cummax()
    drawdown = (portfolio_values - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    return max_drawdown

def calculate_max_one_month_drawdown(portfolio_returns, window=21):
    cumulative_returns = (portfolio_returns + 1).rolling(window).apply(np.prod, raw=True) - 1
    max_one_month_drawdown = cumulative_returns.min()
    return max_one_month_drawdown

# ============================
# 9. Calculate Performance Metrics
# ============================

# Portfolio 1 Metrics
annualized_return_p1, annualized_volatility_p1, sharpe_ratio_p1 = calculate_sharpe_ratio(
    basedata['Portfolio1_Returns']
)
sortino_ratio_p1 = calculate_sortino_ratio(basedata['Portfolio1_Returns'])
alpha_p1, beta_p1 = calculate_alpha_beta(
    basedata['Portfolio1_Returns'], basedata['Portfolio2_Returns']
)
information_ratio, excess_returns = calculate_information_ratio(
    basedata['Portfolio1_Returns'], basedata['Portfolio2_Returns']
)
max_drawdown_p1 = calculate_max_drawdown(basedata['Portfolio1'])
max_one_month_drawdown_p1 = calculate_max_one_month_drawdown(basedata['Portfolio1_Returns'])

# Portfolio 2 Metrics (Benchmark: SPY)
annualized_return_p2, annualized_volatility_p2, sharpe_ratio_p2 = calculate_sharpe_ratio(
    basedata['Portfolio2_Returns']
)
sortino_ratio_p2 = calculate_sortino_ratio(basedata['Portfolio2_Returns'])
max_drawdown_p2 = calculate_max_drawdown(basedata['Portfolio2'])
max_one_month_drawdown_p2 = calculate_max_one_month_drawdown(basedata['Portfolio2_Returns'])

# Compile Metrics into a DataFrame
metrics = {
    'Metric': [
        'Annualized Return', 
        'Annualized Volatility', 
        'Sharpe Ratio', 
        'Sortino Ratio',
        'Information Ratio', 
        'Alpha', 
        'Beta', 
        'Maximum Drawdown',
        'Max 1-Month Drawdown',
        'Initial Balance', 
        'Final Balance'
    ],
    'Portfolio': [
        f"{annualized_return_p1:.2%}", 
        f"{annualized_volatility_p1:.2%}", 
        f"{sharpe_ratio_p1:.2f}", 
        f"{sortino_ratio_p1:.2f}", 
        f"{information_ratio:.2f}",
        f"{alpha_p1:.2%}", 
        f"{beta_p1:.2f}", 
        f"{max_drawdown_p1:.2%}",
        f"{max_one_month_drawdown_p1:.2%}",
        f"${1:.2f}", 
        f"${basedata['Portfolio1'].iloc[-1]:.2f}"
    ],
    'Benchmark (SPY)': [
        f"{annualized_return_p2:.2%}", 
        f"{annualized_volatility_p2:.2%}", 
        f"{sharpe_ratio_p2:.2f}", 
        f"{sortino_ratio_p2:.2f}", 
        '-',  # No Information Ratio for Benchmark
        '-',  # No Alpha and Beta for Benchmark
        '-', 
        f"{max_drawdown_p2:.2%}",
        f"{max_one_month_drawdown_p2:.2%}",
        f"${1:.2f}", 
        f"${basedata['Portfolio2'].iloc[-1]:.2f}"
    ]
}

# Create Metrics DataFrame
metrics_df = pd.DataFrame(metrics)
metrics_df.set_index('Metric', inplace=True)

# ============================
# 10. Display and Save Metrics
# ============================

print("\nPortfolio Performance Metrics:\n")
print(metrics_df.to_string())

# ============================
# 11. Visualization
# ============================

sb.set_theme(style='darkgrid')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.title_fontsize'] = 12

plt.figure(figsize=(12, 6), dpi=100)

# Prepare data for plotting
plot_data = basedata[['Date', 'Portfolio1', 'Portfolio2']].copy()
plot_data = plot_data.melt('Date', var_name='Portfolio', value_name='Normalized Value')

# Rename Portfolio2 for clarity
plot_data['Portfolio'] = plot_data['Portfolio'].replace({'Portfolio2': 'Benchmark: SPY'})

# Plotting
sb.lineplot(data=plot_data, x='Date', y='Normalized Value', hue='Portfolio')

plt.title('Portfolio vs. Benchmark Performance')
plt.xlabel('Date')
plt.ylabel('Normalized Portfolio Value')
plt.legend(title='', loc='upper left')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()

# Save and show plot
plt.savefig('portfolio_performance.png', dpi=300)
plt.show()

# ============================
# 12. Save Weights to CSV
# ============================

# Save selected stocks and weights to CSV (weights in percentages)
selected_stocks = df[df['Tickers'].isin(weightings1.keys())].copy()
selected_stocks['Weights'] = selected_stocks['Weights']  # Already normalized to sum to 100%
selected_stocks[['Tickers', 'Weights']].to_csv('stock_ticker_weights.csv', index=False)

# Verify that weights sum to 100%
total_weight = selected_stocks['Weights'].sum()
print(f"Total Weight (should be 100%): {total_weight:.2f}%")
