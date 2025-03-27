import pandas as pd 
import numpy as np
import multiprocessing as mp

### Resampling ###

def shift_with_sample(
        df:pd.DataFrame | pd.Series, step:int | str = 0
    ) -> pd.DataFrame | pd.Series: 
    """
    Shift the dataframe by a given step after having resampled it daily.
    """
    df_ = df[~df.index.duplicated(keep='first')]
    return df_.resample('1D').ffill().ffill().shift(step)

def shift_ignoring_nan(
    df:pd.DataFrame | pd.Series , step:int | str = 0
) -> pd.DataFrame | pd.Series:
    """
    Shift the dataframe by a given step ignoring the NaN.
    """
    df_ = df[~df.index.duplicated(keep='first')]

    match type(df_):
        case pd.Series:
            return df_.dropna().shift(step)
        case pd.DataFrame:
            return pd.concat({
                col: df_[col].dropna().shift(step)
                for col in df_.columns
            }, axis = 1)
        case _:
            raise TypeError(f"df should be a pd.Series or pd.DataFrame not {type(df_)}")
        
### Align Position with Returns ###

def _align_pos_with_returns_ds(position:pd.DataFrame | pd.Series, returns:pd.Series) -> pd.DataFrame | pd.Series:
    returns_ = returns.dropna()
    return (
        position
        .replace([np.inf, -np.inf], np.nan).ffill()
        .reindex(returns_.index, method='ffill', axis=0).ffill()
        .fillna(0)
    )

def _align_pos_with_returns_df(position:pd.DataFrame, returns:pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        _align_pos_with_returns_ds(position.loc[:, [col]], returns.loc[:, col].dropna())
        for col in returns.columns
    ).ffill().fillna(0)

def align_pos_with_returns(position:pd.DataFrame | pd.Series, returns:pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Align the position with the returns. 
    For each instrument, take a signal and convert it into a position using the returns index. Make sure that: 
    1. There is a position for each of the days in the returns (position has the same index as the returns)
    2. The position can't change on days where the exchange is closed (position is not moved on days where returns are NaN)

    For that, we process as follow: 
    1. If the returns contains only one instrument we apply the following process: 
        a. We take each columns of the return index and get rid of the NaN from the returns (these corresponds to days where the instrument is not traded).
        b. We take the signal and reindex it on the returns index (where the returns are not NaN), and ffill the positions.
    2. If the returns contains multiple instruments, we apply the process to each instrument, concatenate the results, and ffill the positions.
    """

    match (type(position), type(returns)):
        case (pd.DataFrame | pd.Series, pd.Series):
            return _align_pos_with_returns_ds(position, returns.dropna())
        case (pd.DataFrame, pd.DataFrame):
            return _align_pos_with_returns_df(position, returns)
        case _:
            raise ValueError('returns must be a pd.DataFrame or pd.Series')

### Compute Pnl ###

def _compute_pnl_ds(position:pd.DataFrame | pd.Series, returns:pd.Series) -> pd.DataFrame | pd.Series:
    returns_ = returns.dropna()
    pos = _align_pos_with_returns_ds(position, returns_)
    pnl = pos.shift(1).mul(returns_, axis=0)
    assert not pnl.apply(np.isinf).any().any(), 'inf in your pnl'
    return pnl

def _compute_pnl_df(position:pd.DataFrame, returns:pd.DataFrame) -> pd.DataFrame:
    tasks = ( ( position.loc[:, [col]], returns.loc[:, col].dropna() ) for col in returns.columns )
    with mp.Pool(mp.cpu_count() - 2) as pool:
        results = pool.starmap(_compute_pnl_ds, tasks)
    pnl = pd.concat(results, axis = 1)
    return pnl

def compute_pnl(position:pd.DataFrame | pd.Series, returns:pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Compute the pnl from a position and a returns. Position is re-aligned with the returns before computation.
    """
    match (type(position), type(returns)):
        case (pd.Series, pd.Series):
            return _compute_pnl_ds(position, returns.dropna())
        case (pd.DataFrame, pd.DataFrame | pd.Series):
            return _compute_pnl_df(position, returns)
        case _:
            raise ValueError('returns must be a pd.DataFrame or pd.Series')
        