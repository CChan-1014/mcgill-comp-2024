import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  

df = pd.read_excel('Portfolio.xlsx')

weightings1 = dict(zip(df['Tickers'], df['Weights']))
weightings2 = {"SPY": 100}  # Benchmark portfolio

members = list(df['Tickers']) + ["SPY"]

def Backtester(weightings, data, name):
    data[name] = sum([float(weightings[i]) * data[i] / 100 for i in weightings])
    return data

basedata = yf.Ticker(members[0]).history(period="max").reset_index()[["Date", "Open"]]
basedata["Date"] = pd.to_datetime(basedata["Date"])
basedata = basedata.rename(columns={"Open": members[0]})

if len(members) > 1:
    for ticker in members[1:]:
        newdata = yf.Ticker(ticker).history(period="max").reset_index()[["Date", "Open"]]
        newdata["Date"] = pd.to_datetime(newdata["Date"])
        newdata = newdata.rename(columns={"Open": ticker})
        basedata = pd.merge(basedata, newdata, on="Date", how='inner')

# Start date
basedata = basedata[basedata["Date"] > "2010-01-01"]

# Reset index after filtering dates
basedata.reset_index(drop=True, inplace=True)

# Normalise the price data (start with $1)
for ticker in members:
    basedata[ticker] = basedata[ticker] / basedata[ticker].iloc[0]

basedata = Backtester(weightings1, basedata, "Portfolio1")
basedata = Backtester(weightings2, basedata, "Portfolio2")

basedata['Portfolio1_Returns'] = basedata['Portfolio1'].pct_change()
basedata['Portfolio2_Returns'] = basedata['Portfolio2'].pct_change()

basedata = basedata.dropna(subset=['Portfolio1_Returns', 'Portfolio2_Returns'])

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    mean_return = returns.mean()
    std_return = returns.std()
    annualized_return = mean_return * 252  # 252 trading days
    annualized_volatility = std_return * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    return annualized_return, annualized_volatility, sharpe_ratio

annualized_return_p1, annualized_volatility_p1, sharpe_ratio_p1 = calculate_sharpe_ratio(
    basedata['Portfolio1_Returns'], risk_free_rate=0.02
)

annualized_return_p2, annualized_volatility_p2, sharpe_ratio_p2 = calculate_sharpe_ratio(
    basedata['Portfolio2_Returns'], risk_free_rate=0.02
)

print(f"Portfolio1 Annualized Return: {annualized_return_p1:.4%}")
print(f"Portfolio1 Annualized Volatility: {annualized_volatility_p1:.4%}")
print(f"Portfolio1 Sharpe Ratio: {sharpe_ratio_p1:.4f}\n")

print(f"Benchmark Annualized Return: {annualized_return_p2:.4%}")
print(f"Benchmark Annualized Volatility: {annualized_volatility_p2:.4%}")
print(f"Benchmark Sharpe Ratio: {sharpe_ratio_p2:.4f}")


print(basedata)

# Plotting 

plt.plot(basedata["Date"], basedata["Portfolio1"], label="Portfolio1")
plt.plot(basedata["Date"], basedata["Portfolio2"], label="Benchmark (SPY)")

plt.style.use('dark_background')
plt.legend(loc="upper left")
plt.xlabel('Date')
plt.ylabel('Normalized Portfolio Value')
plt.title('Portfolio Performance Comparison')
plt.show()
