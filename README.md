# Kwantlib

A Python library for quantitative analysis and systematic trading.

## Installation

To install the necessary dependencies, use the following command:

```bash
pip install -r requirements.txt
```

## Usage

Here is an example of how to use the library:

```python
import pandas as pd
from kwantlib import Operator, Strategy

# Load price data
price = pd.read_csv('price.csv')

# Calculate returns and volatility
returns = price.pct_change()
vol = returns.rolling(20).std()

# Apply operators
cma = Operator.cross_moving_average(price, smooth_params=[10, 20], lookback_params=[10, 20])
signal = Operator.proj(cma, threshold=0.1)

# Calculate position and PnL
pos = Strategy.compute_position(signal, vol)
pnl = Strategy.compute_pnl(pos, returns)

# Plot cumulative PnL
plotx(pnl.cumsum() / pnl.std())
```

## Backtesting

You can also perform a backtest with the following strategy:

```python
Strategy(
    returns=price.pct_change(),
    signal=price,
    is_vol_target=True
).cross_moving_average().proj().backtest()
```

## Conclusion

Kwantlib is designed to facilitate quantitative analysis and systematic trading. Feel free to explore the various features offered by this library.


