import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('Portfolio.xlsx')

# Create weightings1 dictionary from the DataFrame
weightings1 = dict(zip(df['Tickers'], df['Weights']))
weightings2 = {"SPY": 100}

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

# Specific start date
basedata = basedata[basedata["Date"] > "2010-01-01"]

# Normalize the price data
for ticker in members:
    basedata[ticker] = basedata[ticker] / basedata[ticker].iloc[0]

# Construct the portfolios
basedata = Backtester(weightings1, basedata, "Portfolio1")
basedata = Backtester(weightings2, basedata, "Portfolio2")

# Plot the portfolios
plt.plot(basedata["Date"], basedata["Portfolio1"], label="Portfolio1")
plt.plot(basedata["Date"], basedata["Portfolio2"], label="Portfolio2")

plt.style.use('dark_background')
plt.legend(loc="upper left")
plt.xlabel('Date')
plt.ylabel('Normalized Portfolio Value')
plt.title('Portfolio Performance Comparison')
plt.show()

# Optionally, print the DataFrame to see the data
print(basedata)
