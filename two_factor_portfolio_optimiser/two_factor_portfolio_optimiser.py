import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
tables = pd.read_html(url)
ftse_table = tables[6]  #Could become redundant
tickers = ftse_table['Ticker'].tolist()
tickers = [ticker + ".L" for ticker in tickers]

prices = yf.download(tickers, start='2023-01-01', auto_adjust=True)['Close']
prices = prices.dropna() #removes weekends holidays

momentum = prices.pct_change(63)
volatility = prices.pct_change().rolling(63).std()

latest_momentum = momentum.iloc[-1]
latest_volatility = volatility.iloc[-1]

factors = pd.DataFrame({
    'momentum': latest_momentum,
    'volatility': latest_volatility
})
factors.dropna(inplace=True)

factors['momentum_score'] = (factors['momentum'] - factors['momentum'].mean()) / factors['momentum'].std()
factors['volatility_score'] = - (factors['volatility'] - factors['volatility'].mean()) / factors['volatility'].std()
factors['composite_score'] = (factors['momentum_score'] + factors['volatility_score']) / 2

weights = factors['composite_score']
weights = weights.clip(lower=0)
weights = weights / weights.sum()

monthly_prices = prices.resample('ME').last()
returns = monthly_prices.pct_change().dropna()
portfolio_returns = (returns*weights).sum(axis=1)

benchmark = yf.download('^FTSE', start='2023-01-01')['Close']
benchmark = benchmark.resample('ME').last()
benchmark_returns = benchmark.pct_change().dropna()

cumulative_portfolio = (1 + portfolio_returns).cumprod()
cumulative_benchmark = (1 + benchmark_returns).cumprod()

cumulative_benchmark.plot(label='Benchmark', linewidth=2.5, alpha=0.8, color='blue')
cumulative_portfolio.plot(label='Portfolio', linewidth=2.5, alpha=0.8, color='green')
plt.legend()
plt.ylabel("Return of original investment")
plt.xlabel("Date")
plt.title('Static weight portfolio vs FTSE', fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5) 
plt.tight_layout()
plt.savefig("plot.png")







