import yfinance as yf
import pandas as pd
from datetime import datetime

df = pd.read_csv('stocks.csv')

benchmark_ticker = 'SPY'
start_date = '2000-01-01'
# end_date = datetime.today().strftime('%Y-%m-%d')
end_date = '2020-01-01'

def calculate_alpha_beta(ticker, benchmark_ticker, start_date, end_date):
    try:
        
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date)
        if stock_data.empty or benchmark_data.empty:
            print(f"No data available for {ticker}. Skipping.")
            return None, None
        
        stock_returns = stock_data['Adj Close'].pct_change()
        benchmark_returns = benchmark_data['Adj Close'].pct_change()
        
        returns_df = pd.DataFrame({
            'stock_returns': stock_returns, 
            'benchmark_returns': benchmark_returns
        }).dropna()
        
        if returns_df.empty:
            print(f"Not enough data for {ticker} after aligning dates. Skipping.")
            return None, None

        cov = returns_df['stock_returns'].cov(returns_df['benchmark_returns'])
        var = returns_df['benchmark_returns'].var()
        
        beta = cov / var
        
        stock_mean_return = returns_df['stock_returns'].mean()
        benchmark_mean_return = returns_df['benchmark_returns'].mean()
        alpha = stock_mean_return - beta * benchmark_mean_return
        
        annualized_alpha = alpha * 252 # 252 Trading Days
        
        return annualized_alpha, beta
    except Exception as e:
        print(f"Error processing ticker {ticker}: {e}")
        return None, None

alphas = []
betas = []

for ticker in df['Ticker']:
    alpha, beta = calculate_alpha_beta(ticker, benchmark_ticker, start_date, end_date)
    alphas.append(alpha)
    betas.append(beta)

df['Alpha'] = alphas
df['Beta'] = betas

df['Alpha'] = df['Alpha'].apply(lambda x: f"{x * 100:.2f}%" if pd.notnull(x) else "N/A")
df['Beta'] = df['Beta'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")

print("\nAlpha and Beta of Stocks Against SPY:\n")
print(df.to_string(index=False))
