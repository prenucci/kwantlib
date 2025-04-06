import pandas as pd
import multiprocessing as mp

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

### Tools ###

def _compute_metrics_ds(pnl:pd.Series, pos:pd.Series, pos_change:pd.Series) -> pd.Series:

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
    }, name=pnl.name)

def _compute_metrics_df(pnl:pd.DataFrame, pos:pd.DataFrame, pos_change:pd.DataFrame) -> pd.DataFrame:

    tasks = (
        ( pnl.loc[:, col], pos.loc[:, col], pos_change.loc[:, col]) 
        for col in pnl.columns.intersection(pos.columns).intersection(pos_change.columns)
    )
    with mp.Pool(mp.cpu_count() - 2) as pool:
        results = pool.starmap(_compute_metrics_ds, tasks)

    return pd.concat([
        result_col for result_col in results
    ], axis=1).T.sort_values(by='eff_sharpe', ascending=False)
    
    return pd.concat({
        col:result_col for col, result_col in zip(pos.columns, results)
    }, axis=1).T.sort_values(by='eff_sharpe', ascending=False)

def compute_metrics(
    pnl:pd.DataFrame | pd.Series, 
    pos:pd.DataFrame | pd.Series, 
    pos_change:pd.DataFrame | pd.Series = None
) -> pd.DataFrame | pd.Series:
    
    """ Get the metrics for the (position, pnl, position change) tuple. """
    
    if pos_change is None:
        pos_change = pos.diff().abs() 

    match (type(pos), type(pnl), type(pos_change)):
        case (pd.Series, pd.Series, pd.Series):
            return _compute_metrics_ds(pnl=pnl, pos=pos, pos_change=pos_change)
        case (pd.DataFrame, pd.DataFrame, pd.DataFrame):
            return _compute_metrics_df(pnl=pnl, pos=pos, pos_change=pos_change)
        case _:
            raise ValueError('pos, pnl and pos_change must be of the same type')
    