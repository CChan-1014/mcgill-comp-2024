# import pandas as pd
# from sklearn.linear_model import ElasticNet
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error

# # Load the updated testdata and fama-french data
# famafrench_df = pd.read_csv('famafrench.csv')
# testdata_df = pd.read_csv('testdata.csv')

# # Ensure the 'date' column is processed properly
# testdata_df['date'] = testdata_df['date'].astype(str).str[:6]
# famafrench_df['date'] = famafrench_df['date'].astype(str)

# # Merge the datasets on 'date'
# merged_df = pd.merge(testdata_df, famafrench_df, on='date')

# # Define the stock tickers for the "Magnificent 7"
# magnificent_7 = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA']

# # Initialize scaler for Z-score normalization
# scaler = StandardScaler()

# # Initialize a list to store results
# results_zscore = []

# # Iterate over each stock ticker and run elastic net regression
# for stock in magnificent_7:
#     stock_data = merged_df[merged_df['stock_ticker'] == stock]

# # # Get the list of all unique stock tickers from the dataset
# # all_tickers = merged_df['stock_ticker'].unique()

# # # Iterate over each stock ticker and run elastic net regression
# # for stock in all_tickers:
# #     stock_data = merged_df[merged_df['stock_ticker'] == stock]

    
#     # Set the predictors (Fama-French factors) and the response variable (stock excess return)
#     X = stock_data[['SMB', 'HML', 'RMW', 'CMA', 'RF']]
#     y = stock_data['stock_exret'].astype(float)
    
#     # Apply Z-score normalization to the features
#     X_zscore_scaled = scaler.fit_transform(X)
    
#     # Split the data into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X_zscore_scaled, y, test_size=0.3, random_state=42)
    
#     # Initialize and fit the ElasticNet model with a lower alpha value
#     elastic_net = ElasticNet(alpha=0.00001, l1_ratio=0.5, random_state=42)
#     elastic_net.fit(X_train, y_train)
    
#     # Extract coefficients (Betas)
#     betas = elastic_net.coef_
    
#     # Append the result for the stock ticker
#     results_zscore.append({
#         'Stock Ticker': stock,
#         'Beta_1 (SMB)': betas[0],
#         'Beta_2 (HML)': betas[1],
#         'Beta_3 (RMW)': betas[2],
#         'Beta_4 (CMA)': betas[3],
#         'Beta_5 (RF)': betas[4]
#     })

# # Convert results into a DataFrame and output the results
# results_df = pd.DataFrame(results_zscore)
# print(results_df)
# print(merged_df['date'])

import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the updated testdata and fama-french data
famafrench_df = pd.read_csv('/Users/Ray.Fang/Downloads/Elastic Net Regression (Fama French)/famafrench.csv')
testdata_df = pd.read_csv('/Users/Ray.Fang/Downloads/Elastic Net Regression (Fama French)/testdata.csv')

# Ensure the 'date' column is processed properly
testdata_df['date'] = testdata_df['date'].astype(str).str[:6]
famafrench_df['date'] = famafrench_df['date'].astype(str)

# Merge the stock excess returns with the Fama-French data on 'date'
merged_df = pd.merge(testdata_df, famafrench_df, on='date')

filtered_df = merged_df[(merged_df['date'] >= '200001') & (merged_df['date'] <= '200912')]

# Drop rows where stock excess return is missing (but not the Fama-French factors)
# merged_df = merged_df.dropna(subset=['stock_exret'])
filtered_df = filtered_df.dropna(subset=['stock_exret'])

# Get the list of all unique stock tickers from the cleaned dataset
# all_tickers = merged_df['stock_ticker'].unique()
all_tickers = filtered_df['stock_ticker'].unique()

# Initialize scaler for Z-score normalization
scaler = StandardScaler()

# Initialize a list to store results
results_zscore = []

# Iterate over each stock ticker and run elastic net regression
for stock in all_tickers:
    # stock_data = merged_df[merged_df['stock_ticker'] == stock]
    stock_data = filtered_df[filtered_df['stock_ticker'] == stock]
    
    # Ensure there is sufficient data (e.g., at least 6 data points for a meaningful regression)
    if len(stock_data) < 6:
        continue  # Skip stocks with insufficient data
    
    # Set the predictors (Fama-French factors) and the response variable (stock excess return)
    X = stock_data[['SMB', 'HML', 'RMW', 'CMA', 'RF']]
    y = stock_data['stock_exret'].astype(float)
    
    # Check for constant or low-variation in response variable y
    if len(np.unique(y)) < 2:
        continue  # Skip this stock due to lack of variation in y
    
    # Apply Z-score normalization to the features
    X_zscore_scaled = scaler.fit_transform(X)
    
    # Optionally scale the response variable (y) to improve convergence
    # y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
    
    # Perform the train-test split if there are more than 1 sample
    if len(stock_data) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X_zscore_scaled, y, test_size=0.3, random_state=42)
    else:
        X_train, y_train = X_zscore_scaled, y
    
    # Initialize and fit the ElasticNet model with more regularization and more iterations
    elastic_net = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42, max_iter=20000, tol=0.001)
    elastic_net.fit(X_train, y_train)
    print("Enet fitting works")
    
    # Extract coefficients (Betas)
    betas = elastic_net.coef_
    
    # Append the result for the stock ticker
    results_zscore.append({
        'Stock Ticker': stock,
        'Beta_1 (SMB)': betas[0],
        'Beta_2 (HML)': betas[1],
        'Beta_3 (RMW)': betas[2],
        'Beta_4 (CMA)': betas[3],
        'Beta_5 (RF)': betas[4]
    })

# Convert results into a DataFrame and output the results
results_df = pd.DataFrame(results_zscore)
print(results_df)
results_df.to_csv('elastic_net_coefficients_before2010.csv', index=False)

