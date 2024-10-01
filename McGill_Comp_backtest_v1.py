import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

weightings1 = {"MSFT": "50", "NVDA": "50"}
weightings2 = {"SPY": "100"}

members = ["MSFT", "NVDA", "SPY"]

def Backtester(weigtings, data, name):
    #Construct portfolio
    data[name] = sum([int(weigtings[i])*data[i]/100 for i in list(weigtings.keys())])
    return data

basedata = yf.Ticker(members[0]).history(period="max").reset_index()[["Date", "Open"]]
basedata["Date"] = pd.to_datetime(basedata["Date"])
basedata = basedata.rename(columns = {"Open": members[0]})
if len(members) > 1:
    for i in range(1, len(members)):
        newdata = yf.Ticker(members[i]).history(period="max").reset_index()[["Date", "Open"]]
        newdata["Date"] = pd.to_datetime(newdata["Date"])
        newdata = newdata.rename(columns = {"Open": members[i]})
        basedata = pd.merge(basedata, newdata, on="Date")

basedata = basedata[  basedata["Date"] > "2010-01-01"]

#Normalize the data
for i in members:
    basedata[i] = basedata[i]/basedata[i].iloc[0]

basedata = Backtester(weightings1, basedata, "Portfolio1")
basedata = Backtester(weightings2, basedata, "Portfolio2")

plt.plot(basedata["Date"], basedata["Portfolio1"], label="Portfolio1")
plt.plot(basedata["Date"], basedata["Portfolio2"], label="Portfolio2")

plt.style.use('dark_background')
plt.legend(loc = "upper left")
plt.show()

print(basedata)