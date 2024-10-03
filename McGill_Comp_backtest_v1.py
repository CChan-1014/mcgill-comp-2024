import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from matplotlib.ticker import PercentFormatter

df = pd.read_csv('Portfolio2.csv')

weightings1 = dict(zip(df['Tickers'], df['Weights']))
weightings2 = {"SPY": 100}  # Benchmark

members = list(df['Tickers']) + ["SPY"]

def Backtester(weightings, data, name):
    data[name] = sum([float(weightings[i]) * data[i] / 100 for i in weightings])
    return data

basedata = yf.Ticker(members[0]).history(period="max").reset_index()[["Date", "Close"]]
basedata["Date"] = pd.to_datetime(basedata["Date"])
basedata = basedata.rename(columns={"Close": members[0]})

if len(members) > 1:
    for ticker in members[1:]:
        newdata = yf.Ticker(ticker).history(period="max").reset_index()[["Date", "Close"]]
        newdata["Date"] = pd.to_datetime(newdata["Date"])
        newdata = newdata.rename(columns={"Close": ticker})
        basedata = pd.merge(basedata, newdata, on="Date", how='inner')

# Start date
basedata = basedata[basedata["Date"] > "2000-01-01"]
basedata.reset_index(drop=True, inplace=True)

# Normalize the price data (start with $1)
for ticker in members:
    try:
        basedata[ticker] = basedata[ticker] / basedata[ticker].iloc[0]
    except KeyError:
        pass

basedata = Backtester(weightings1, basedata, "Portfolio1")
basedata = Backtester(weightings2, basedata, "Portfolio2")
basedata['Portfolio1_Returns'] = basedata['Portfolio1'].pct_change()
basedata['Portfolio2_Returns'] = basedata['Portfolio2'].pct_change()
basedata = basedata.dropna(subset=['Portfolio1_Returns', 'Portfolio2_Returns'])

# Calculations:

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

annualized_return_p2, annualized_volatility_p2, sharpe_ratio_p2 = calculate_sharpe_ratio(
    basedata['Portfolio2_Returns']
)
sortino_ratio_p2 = calculate_sortino_ratio(basedata['Portfolio2_Returns'])
max_drawdown_p2 = calculate_max_drawdown(basedata['Portfolio2'])
max_one_month_drawdown_p2 = calculate_max_one_month_drawdown(basedata['Portfolio2_Returns'])

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

# Data output
metrics_df = pd.DataFrame(metrics)
metrics_df.set_index('Metric', inplace=True)

print("\nPortfolio Performance Metrics:\n")
print(metrics_df.to_string())

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

# Melt the data
plot_data = basedata[['Date', 'Portfolio1', 'Portfolio2']].copy()
plot_data = plot_data.melt('Date', var_name='Portfolio', value_name='Normalized Value')

plot_data['Portfolio'] = plot_data['Portfolio'].replace({'Portfolio2': 'Benchmark: SPY'})

sb.lineplot(data=plot_data, x='Date', y='Normalized Value', hue='Portfolio')

plt.title('Portfolio vs. Benchmark Performance')
plt.xlabel('Date')
plt.ylabel('Normalized Portfolio Value')
plt.legend(title='', loc='upper left')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()

plt.savefig('portfolio_performance.png', dpi=300)
plt.show()
