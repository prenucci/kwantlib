import pandas as pd 
import multiprocessing as mp 
import plotly.express as px
from .pnl_tools import align_pos_with_returns, compute_pnl

### Plot with plotly ###

def plotx(df:pd.Series | pd.DataFrame, title:str = None):
    """
    Plot the dataframe using plotly.express
    """

    match type(df):
        case pd.Series:
            return px.line(df, title=title)
        case pd.DataFrame:
            df_ = df.copy() 
            if isinstance(df_, pd.DataFrame) and isinstance(df_.columns, pd.MultiIndex):
                df_.columns = df_.columns.map(lambda x: ''.join(str(u) for u in x))
            return px.line(df_, title=title)
        case _:
            raise TypeError
    
### Metrics ###

def sharpe(pnl:pd.DataFrame | pd.Series) -> pd.Series | float:
    return 16 * pnl.mean() / pnl.std()

def turnover(
    pos:pd.DataFrame | pd.Series, pos_change:pd.DataFrame | pd.Series
) -> pd.Series | float:
    return pos_change.mean() / pos.abs().mean() 

def pnl_per_trade(
    pnl:pd.DataFrame | pd.Series, pos_change:pd.DataFrame | pd.Series
) -> pd.Series | float:
    return pnl.mean() / pos_change.mean()

def maxdrawdown(pnl:pd.DataFrame | pd.Series) -> pd.Series | float:
    return (
        pnl.cumsum().cummax() - pnl.cumsum()
    ).max() / ( pnl.std() * 16 )

def calamar(pnl:pd.DataFrame | pd.Series) -> pd.Series | float:
    return sharpe(pnl) / maxdrawdown(pnl)

def ftrading(pos:pd.DataFrame | pd.Series) -> pd.Series | float:
    return (pos.abs() != 0).mean()

def sortino(pnl:pd.DataFrame | pd.Series) -> pd.Series | float:
    return 16 * pnl.mean() / pnl[pnl < 0].std()

def loss_std_ratio(pnl:pd.DataFrame | pd.Series) -> pd.Series | float:
    return pnl[pnl < 0].std() / pnl[pnl != 0].std()

def win_rate(pnl:pd.DataFrame | pd.Series) -> pd.Series | float:
    return (pnl > 0).sum() / (pnl != 0).sum() 

def unlevered_mean_return(pnl:pd.DataFrame | pd.Series, pos:pd.DataFrame | pd.Series) -> pd.Series | float:
    return pnl.mean() / pos.abs().mean()

def unlevered_std(pnl:pd.DataFrame | pd.Series, pos:pd.DataFrame | pd.Series) -> pd.Series | float:
    return pnl.std() / pos.abs().mean()

### Metrics global ###

def _metrics_ds(pos:pd.Series, pnl:pd.Series, pos_change:pd.Series) -> pd.Series:
    return pd.Series({
        'eff_sharpe': sharpe(pnl[pnl!=0]),
        'raw_sharpe': sharpe(pnl),
        'turnover (%)': 100 * turnover(pos, pos_change),
        'pnl_per_trade (bps)': 1e4 * pnl_per_trade(pnl, pos_change),
        'unlev_return (y%)': 100 * 252 * unlevered_mean_return(pnl, pos),
        'unlev_std (y%)': 100 * 16 * unlevered_std(pnl, pos),
        'maxdrawdown (ystd)': maxdrawdown(pnl),
        'ftrading (%)': 100 * ftrading(pos),
        'calamar': calamar(pnl),
        'sortino': sortino(pnl),
        'loss_std_ratio (%)': 100 * loss_std_ratio(pnl),
        'win_rate (%)': 100 * win_rate(pnl),
        'r_sharpe': sharpe(pnl.fillna(0).rolling(252).mean()) / 16,
    })

def _metrics_df(pos:pd.DataFrame, pnl:pd.DataFrame, pos_change:pd.DataFrame) -> pd.DataFrame:
    tasks = (
        ( pos.loc[:, col], pnl.loc[:, col].fillna(0), pos_change.loc[:, col] ) 
        for col in pos.columns
    )
    with mp.Pool(mp.cpu_count() - 2) as pool:
        results = pool.starmap(_metrics_ds, tasks)
    
    return pd.concat({
        col:result_col for col, result_col in zip(pos.columns, results)
    }, axis=1).T.sort_values(by='eff_sharpe', ascending=False)

def resample_daily(
    pnl:pd.DataFrame | pd.Series, pos_abs:pd.DataFrame | pd.Series, pos_change:pd.DataFrame | pd.Series
) -> tuple[pd.DataFrame | pd.Series, pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]:
    
    """
    Resample the pnl, pos_abs and pos_change to daily when intraday. 
    """
    
    match (type(pos_abs), type(pnl), type(pos_change)):
        case (pd.Series, pd.Series, pd.Series):
            pass
        case (pd.DataFrame, pd.DataFrame, pd.DataFrame):
            instruments = pnl.columns.intersection(pos_abs.columns).intersection(pos_change.columns)
            pnl, pos_abs, pos_change = pnl.loc[:, instruments], pos_abs.loc[:, instruments], pos_change.loc[:, instruments]
        case _:
            raise ValueError('pos_abs, pnl and pos_change must be of the same type')
    
    match (hasattr(pos_abs.index, 'date'), hasattr(pnl.index, 'date'), hasattr(pos_change.index, 'date')):
        case (True, True, True):
            pos_abs = pos_abs.ffill().groupby(pos_abs.index.date).mean()
            pnl = pnl.fillna(0).groupby(pnl.index.date).sum()
            pos_change = pos_change.fillna(0).groupby(pos_change.index.date).sum()
        case (False, False, False):
            pass
        case _:
            raise ValueError('pos_abs, pnl and pos_change must be either all daily or all intraday')

    return pnl, pos_abs, pos_change

def metrics(
    pnl:pd.DataFrame | pd.Series, pos:pd.DataFrame | pd.Series, pos_change:pd.DataFrame | pd.Series = None
) -> pd.DataFrame | pd.Series:
    
    """Get the metrics for the (position, pnl, position change) tuple."""
    
    if pos_change is None:
        pos_change = pos.diff().abs() 

    pos_abs, pnl, pos_change = resample_daily(pos.abs(), pnl, pos_change)

    match (type(pos_abs), type(pnl), type(pos_change)):
        case (pd.Series, pd.Series, pd.Series):
            return _metrics_ds(pos_abs, pnl, pos_change)
        case (pd.DataFrame, pd.DataFrame, pd.DataFrame):
            return _metrics_df(pos_abs, pnl, pos_change)
        case _:
            raise ValueError('pos, pnl and pos_change must be of the same type')
    
def _rolling_sharpe(pnl:pd.Series, periods:list[int] = [1/2, 1, 2, 4, 8, 10, 15]) -> pd.DataFrame:
    return pd.concat({
        f'{n}D': 16 * pnl.rolling(n).mean() / pnl.rolling(n).std()
        for n in [int(252 * x) for x in periods]
    } | {
        'expanding': 16 * pnl.expanding(min_periods=1008).mean() / pnl.expanding(min_periods=1008).std(),
        'overall': pd.Series(16 * pnl.mean() / pnl.std(), index=pnl.index), 
    }, axis=1).ffill()

def _pnl_cum(pnl:pd.Series, risk:float = 1, is_aum_cum:bool = False) -> pd.Series:
    pnl_scaled = (risk / 16) * ( pnl / pnl.std() )
    if is_aum_cum:
        return (1 + ( pnl_scaled / 100 )).cumprod()
    else:
        return pnl_scaled.cumsum()

def _drawdown(pnl:pd.Series, risk:float = 1, is_aum_cum:bool = False) -> pd.Series:
    pnl_cum_ = _pnl_cum(pnl, risk, is_aum_cum)
    if is_aum_cum:
        return ( pnl_cum_ - pnl_cum_.cummax() ) / pnl_cum_.cummax()
    else:
        return ( pnl_cum_ - pnl_cum_.cummax() )

### Backtest ###

def backtest(
    pnl:pd.DataFrame, 
    pos:pd.DataFrame = None, 
    pos_change:pd.DataFrame = None, 
    risk:float = 1, is_aum_cum:bool = False
) -> pd.DataFrame:   
    
    """
    Backtest the strategy using the pnl, position and position change.
    Gives a global view (aggregated pnl for all instruments) of the strategy and a per-instrument view.
    """
    
    if pos is None:
        pos = pd.DataFrame(1, index=pnl.index, columns=pnl.columns)
    
    if pos_change is None:
        pos_change = pos.diff().abs()

    pos_abs, pnl, pos_change = resample_daily(pos.abs(), pnl, pos_change)
    pos_abs_total, pnl_total, pos_change_total = pos_abs.sum(1), pnl.sum(1), pos_change.sum(1)
    
    print(metrics(pos_abs_total, pnl_total, pos_change_total).to_frame('overall').T)

    px.line(_pnl_cum(pnl_total, risk, is_aum_cum), title='Pnl cum', log_y= is_aum_cum).show()
    px.line(_drawdown(pnl_total, risk, is_aum_cum), title='drawdown').show()
    px.line(_rolling_sharpe(pnl_total), title='rolling sharpe').show()

    if len(pnl.columns) > 1:
        px.imshow(pnl.corr().fillna(0)).show()

    if len(pnl.columns) < 70:
        plotx( risk * pnl.cumsum() / pnl.std(), title='pnl decomposed' ).show()

    return pd.concat([
        metrics(pos_abs, pnl, pos_change),
        pnl.corrwith(pnl_total).to_frame('corr_with_book')
    ], axis=1)

def quick_backtest(signal:pd.DataFrame, returns:pd.DataFrame, risk:float = 1, is_aum_cum:bool = False) -> pd.DataFrame:
    """
    Quick backtest the strategy using the signal and the returns.
    Signal is first aligned with the returns before backtesting. 
    """
    pos = align_pos_with_returns(signal, returns)
    pos_change = pos.diff().abs()
    pnl = compute_pnl(pos, returns)
    return backtest(pnl, pos, pos_change, risk, is_aum_cum)


