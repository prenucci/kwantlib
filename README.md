# Kwantlib

A Python library for quantitative analysis and systematic trading.


## Usage

Here is an example of how to use the library:

```python
import pandas as pd
import kwantlib as kl

# Load price data
price = pd.read_csv('price.csv')

# Calculate returns and volatility
returns = price.pct_change()
vol = returns.rolling(20).std()

# Apply operators
signal = kl.cross_moving_average(price, smooth_params=[1, 10], lookback_params=[100, 200])

# Calculate position and PnL
pos = kl.compute_position(signal, returns, shift=1)
pnl = kl.compute_pnl(pos, returns)

kl.backtest(pos, pnl)
```

## Conclusion

Kwantlib is designed to facilitate quantitative analysis and systematic trading. Feel free to explore the various features offered by this library.


