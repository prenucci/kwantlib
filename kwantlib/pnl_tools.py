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
    df_ = df[~df.index.duplicated(keep='first')].sort_index(axis=0)
    return df_.resample('1D').ffill().ffill().shift(step)

def shift_ignoring_nan(
    df:pd.DataFrame | pd.Series , step:int | str = 0
) -> pd.DataFrame | pd.Series:
    """
    Shift the dataframe by a given step ignoring the NaN.
    """
    df_ = df[~df.index.duplicated(keep='first')].sort_index(axis=0)

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
        if col in position.columns.get_level_values(0)
    ).ffill().fillna(0)

def align_pos_with_returns(position:pd.DataFrame | pd.Series, returns:pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Align the position with the returns. For each instrument, take a signal and convert it into a position using the returns index. 

    The difficulty is that the returns can contain several instruments for which exchanges does not necessarily have the same trading days. 
    NaN are supposed to be days where the instrument is not traded.
    
    We want to ensure that: 
    1. There is a position for each of the days where the instrument is traded (position has the same index as the returns without NaN)
    2. The position can't change on days where the exchange is closed (position is not moved on days where returns are NaN)
    3. The position is computed from the latest available information (position is ffilled properly)

    For that, we process as follow: 
    1. If the returns contains only one instrument we take the returns, drop the NaN, reindex the position to match the NotNa index of the returns and forward fill.
    2. If the returns contains multiple instruments, we apply the process to each instrument, concatenate the results, and ffill the positions.
    """

    match (type(position), type(returns)):
        case (pd.DataFrame | pd.Series, pd.Series):
            return _align_pos_with_returns_ds(position, returns)
        case (pd.DataFrame, pd.DataFrame):
            return _align_pos_with_returns_df(position, returns)
        case _:
            raise ValueError('returns must be a pd.DataFrame or pd.Series')

### Compute Pnl ###

def _compute_pnl_ds(position:pd.DataFrame | pd.Series, returns:pd.Series) -> pd.DataFrame | pd.Series:
    returns_ = returns.dropna()
    pos_shifted = _align_pos_with_returns_ds(position.shift(1), returns_)
    pnl = pos_shifted.mul(returns_, axis=0)
    assert not pnl.isna().any().any(), 'NaN in your pnl'
    assert not pnl.apply(np.isinf).any().any(), 'inf in your pnl'
    return pnl

def _compute_pnl_df(position:pd.DataFrame, returns:pd.DataFrame) -> pd.DataFrame:
    tasks = ( 
        ( position.loc[:, [col]], returns.loc[:, col].dropna() ) 
        for col in returns.columns 
        if col in position.columns.get_level_values(0)
    )
    with mp.Pool(mp.cpu_count() - 2) as pool:
        results = pool.starmap(_compute_pnl_ds, tasks)
    pnl = pd.concat(results, axis = 1)
    return pnl.fillna(0)

def compute_pnl(position:pd.DataFrame | pd.Series, returns:pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Compute the pnl from a position and a returns. Position is re-aligned with the returns before computation.
    """
    
    if missing_position := [col for col in returns.columns if col not in position.columns.get_level_values(0)]:
        print(f'missing position: {missing_position}')
    if missing_returns := [col for col in position.columns.get_level_values(0) if col not in returns.columns]:
        print(f'missing returns: {missing_returns}')

    match (type(position), type(returns)):
        case (pd.Series, pd.Series):
            pnl = _compute_pnl_ds(position, returns)
        case (pd.DataFrame, pd.DataFrame | pd.Series):
            pnl = _compute_pnl_df(position, returns)
        case _:
            raise ValueError('returns must be a pd.DataFrame or pd.Series')
    return pnl.fillna(0)