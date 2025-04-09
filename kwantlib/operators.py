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

    signal_ = signal.dropna()

    window_params = set(x for x in smooth_params + lookback_params)
    
    mov_avg = {
        window: signal_.ewm(halflife=window).mean() if is_ewm else signal_.rolling(window).mean()
        for window in window_params
    }

    mov_std = {
        window: signal_.ewm(halflife=window).std() if is_ewm else signal_.rolling(window).std()
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
    Compute the cross moving average of the signal df for different smooth and lookback parameters. (Multiprocessed by columns)

    cma(smooth, lookback) = (mov_avg(smooth) - mov_avg(lookback)) / mov_std(lookback)

    - is_proj is a boolean that indicates if the output signal should be averaged to keep the same dimension as the input signal.
    if is_proj is True, the output signal is the average of the cross moving average for all parameters. 
    if is_proj is False, the output signal is a MultiIndex dataframe adding 2 new levels for smooth and lookback.

    - is_ewm is a boolean that indicates if the moving average and std should be computed using an exponential moving average.
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
        
def zscore(
        signal:pd.DataFrame | pd.Series, lookback:int = 1008, is_ewm:bool = False
    ) -> pd.DataFrame | pd.Series:
    """
    Compute the zscore of the signal.
    """
    return cross_moving_average(
        signal, 
        smooth_params = (1,), lookback_params = (lookback,), 
        is_proj = True, is_ewm = is_ewm
    )

def _clip_via_zscore_ds(
        signal:pd.Series, bound:float, lookback:int, clip_inbound:bool, is_ewm:bool
    ) -> pd.Series:

    signal_ = signal.dropna()

    grouper = signal_.ewm(lookback) if is_ewm else signal_.rolling(lookback)
    upper_bound = grouper.mean() + bound * grouper.std()
    lower_bound = grouper.mean() - bound * grouper.std()    

    return (
        signal_
        .mask(signal_ > upper_bound, upper_bound)
        .mask(signal_ < lower_bound, lower_bound)
        .ffill().fillna(0)
    ) if not clip_inbound else (
        signal_
        .where((signal_ > upper_bound) | (signal_ < lower_bound), 0)
        .ffill().fillna(0)
    )

def _clip_via_zscore_df(
        signal:pd.DataFrame, bound:float, lookback:int, clip_inbound:bool, is_ewm:bool
    ) -> pd.DataFrame:

    return pd.concat({
        col: _clip_via_zscore_ds(
            signal=signal.loc[:, col], bound=bound, lookback=lookback, clip_inbound=clip_inbound, is_ewm=is_ewm
        ) 
        for col in signal.columns
    }, axis = 1)

def clip_via_zscore(
        signal:pd.DataFrame | pd.Series, 
        bound:float = 5, lookback:int = 1008, 
        clip_inbound:bool=False, is_ewm:bool = True
    ) -> pd.DataFrame | pd.Series:
    """
    Clip the signal depending on its zscore.
    Clip outbound at the bound value when the signal is greater than the bound and clip_inbound=False (Default)
    Clip inbound at 0 when the signal is lesser than the bound and clip_inbound=True
    """

    match type(signal):
        case pd.Series:
            return _clip_via_zscore_ds(signal=signal, bound=bound, lookback=lookback, clip_inbound=clip_inbound, is_ewm=is_ewm)
        case pd.DataFrame:
            return _clip_via_zscore_df(signal=signal, bound=bound, lookback=lookback, clip_inbound=clip_inbound, is_ewm=is_ewm)
        case _:
            raise ValueError(f"df should be a pd.Series or pd.DataFrame not {type(signal)}")
        
def _seasonal_zscore_ds(signal:pd.Series, lookback:int = 3000) -> pd.Series:
    signal_ = signal.dropna()
    group = signal_.groupby(signal_.index.dayofyear).ewm(lookback)

    mean_over_dayofyear = group.mean()
    mean_over_dayofyear.index = mean_over_dayofyear.index.droplevel(0)
    mean_over_dayofyear = mean_over_dayofyear.sort_index(axis=0)

    std_over_dayofyear = group.std()
    std_over_dayofyear.index = std_over_dayofyear.index.droplevel(0)
    std_over_dayofyear = std_over_dayofyear.sort_index(axis=0)

    return (signal_ - mean_over_dayofyear) / std_over_dayofyear

def _seasonal_zscore_df(signal:pd.DataFrame, lookback:int = 3000) -> pd.DataFrame:
    return pd.concat({
        col:_seasonal_zscore_ds(signal[col].dropna(), lookback)
        for col in signal.columns
    }, axis=1)

def seasonal_zscore(signal:pd.DataFrame | pd.Series, lookback:int = 3000) -> pd.DataFrame | pd.Series:
    match type(signal):
        case pd.Series():
            return _seasonal_zscore_ds(signal, lookback)
        case pd.DataFrame():
            return _seasonal_zscore_df(signal, lookback)
        case _:
            raise ValueError(f"Invalid signal type: {type(signal)}")