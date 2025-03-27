import pandas as pd
import numpy as np
import multiprocessing as mp
from typing import Iterable

def _cross_moving_average_ds(
        signal:pd.Series, 
        smooth_params:Iterable[int], 
        lookback_params:Iterable[int], 
        is_proj:bool, is_ewm:bool,
    ) -> pd.DataFrame:

    window_params = set(x for x in smooth_params + lookback_params)

    mov_avg = {
        window: signal.ewm(halflife=window).mean() if is_ewm else signal.rolling(window).mean()
        for window in window_params
    }

    mov_std = {
        window: signal.ewm(halflife=window).std() if is_ewm else signal.rolling(window).std()
        for window in lookback_params
    }

    new_signal = pd.concat({
        (f'{smooth}', f'{lookback}'): ( mov_avg[smooth] - mov_avg[lookback] ) / mov_std[lookback] 
        for smooth in smooth_params
        for lookback in lookback_params
        if smooth * 2 <= lookback
    }, axis = 1)

    del mov_avg, mov_std

    new_signal = new_signal.where(new_signal.abs() < 5, new_signal.apply(np.sign) * 5)

    return new_signal.mean(1) if is_proj else new_signal

def _cross_moving_average_df(
        signal:pd.DataFrame, 
        smooth_params:Iterable[int], 
        lookback_params:Iterable[int], 
        is_proj:bool, is_ewm:bool, 
    ) -> pd.DataFrame:

    tasks = ( 
        (
            signal.loc[:, col].dropna(), smooth_params, lookback_params, is_proj, is_ewm
        ) 
        for col in signal.columns
    )
    with mp.Pool(mp.cpu_count() - 2) as pool:
        results = pool.starmap(_cross_moving_average_ds, tasks)
        
    return pd.concat({
        col: res for col, res in zip(signal.columns, results)
    }, axis=1)

def cross_moving_average(        
        signal:pd.DataFrame | pd.Series, 
        smooth_params:Iterable[int] = (1, 2, 3, 4, 6, 8, 10,), 
        lookback_params:Iterable[int] = (2, 3, 4, 5, 6, 10, 12, 14, 17, 20, 28, 36, 44, 66,), 
        is_proj:bool = True, is_ewm:bool = False, 
    ) -> pd.DataFrame: 

    """
    Compute the cross moving average of the signal df for different smooth and lookback parameters. 
    The functions is multiprocessed by columns.
    """

    match type(signal):
        case pd.Series:
            return _cross_moving_average_ds(
                signal, smooth_params, lookback_params, is_proj, is_ewm
            )
        case pd.DataFrame:
            return _cross_moving_average_df(
                signal, smooth_params, lookback_params, is_proj, is_ewm,  
            )
        case _:
            raise ValueError(f"signal should be a pd.Series or pd.DataFrame not {type(signal)}")
        
def zscore(signal:pd.DataFrame | pd.Series, lookback:int = 1008, is_ewm:bool = False) -> pd.DataFrame | pd.Series:
    """
    Compute the zscore of the signal.
    """
    return cross_moving_average(
        signal, 
        smooth_params = (1,), lookback_params = (lookback,), 
        is_proj = True, is_ewm = is_ewm
    )

def _clip_via_zscore_ds(ds:pd.Series, bound:float, lookback:int, is_ewm:bool) -> pd.Series:

    ds_ = ds.replace([np.inf, -np.inf], np.nan).dropna()

    grouper = ds_.ewm(lookback) if is_ewm else ds_.rolling(lookback)
    upper_bound = grouper.mean() + bound * grouper.std()
    lower_bound = grouper.mean() - bound * grouper.std()    

    return (
        ds_
        .mask(ds_ > upper_bound, upper_bound)
        .mask(ds_ < lower_bound, lower_bound)
        .ffill().fillna(0)
    )

def _clip_via_zscore_df(df:pd.DataFrame, bound:float, lookback:int, is_ewm:bool) -> pd.DataFrame:
    """
    Clip the signal for every column of the dataframe.
    """
    return pd.concat({
        col: _clip_via_zscore_ds(df.loc[:, col], bound, lookback, is_ewm) for col in df.columns
    }, axis = 1)

def clip_via_zscore(df:pd.DataFrame | pd.Series, bound:float = 5, lookback:int = 1008, is_ewm:bool = True) -> pd.DataFrame | pd.Series:
    """
    Clip the signal when its zscore is greater than the bound. Clip at the bound value. 
    """

    match type(df):
        case pd.Series:
            return _clip_via_zscore_ds(df, bound, lookback, is_ewm)
        case pd.DataFrame:
            return _clip_via_zscore_df(df, bound, lookback, is_ewm)
        case _:
            raise ValueError(f"df should be a pd.Series or pd.DataFrame not {type(df)}")