import pandas as pd 
import numpy  as np 
import multiprocessing as mp 
from typing import Iterable, List, Literal
from functools import cache 

from sklearn.covariance import LedoitWolf
from .pandasMonkeypatch import apply_pd_monkeypatch
from sklearn.linear_model import ElasticNet
from tskit_learn.timeseriesmodel import ExpandingModel
from tskit_learn.utilitaires import expanding_decorator

apply_pd_monkeypatch()

############ Operators

def proj(
    signal:pd.DataFrame, 
    threshold:float=0, 
    level:int | List[int] = 0, 
) -> pd.DataFrame: 
    """
    Reduce the dimensionality of the signal df by averaging over the specified levels. 
    """

    new_signal = (
        signal
        .where(signal.abs() >= threshold, 0)
        .groupby(level = level, axis = 1)
        .mean()
    ) if level != -1 else signal.mean(1).to_frame('projected')

    return new_signal.where(new_signal.abs() >= threshold, 0)

def vote(signal:pd.DataFrame, threshold:float=0.7, level:int | List[int] = 0, ) -> pd.DataFrame: 
    """
    Reduce the dimensionality of the signal df by voting the different signals over the specified levels. 
    """
    return proj(signal.apply(np.sign), threshold = threshold, level = level)


def _moving_average_static(signal:pd.DataFrame, windows:int, is_ewm:bool = False) -> pd.DataFrame:
    return signal.apply(lambda x:x.dropna().ewm(halflife=windows).mean()) if is_ewm else signal.apply(lambda x:x.dropna().rolling(windows).mean())

def _moving_std_static(signal:pd.DataFrame, windows:int, is_ewm:bool = False) -> pd.DataFrame:
    return signal.apply(lambda x:x.dropna().ewm(halflife=windows).std()) if is_ewm else signal.apply(lambda x:x.dropna().rolling(windows).std())

def cross_moving_average(        
        signal:pd.DataFrame, 
        smooth_params:Iterable[int] = (1, 2, 3, 4, 6, 8, 10,), 
        lookback_params:Iterable[int] = (2, 3, 4, 5, 6, 10, 12, 14, 17, 20, 28, 36, 44, 66,), 
        is_ewm:bool = False,
        n_jobs:int = mp.cpu_count() - 2,
    ) -> pd.DataFrame: 

    window_params = set(x for x in smooth_params + lookback_params)

    tasks_ma = ( (signal, window, is_ewm) for window in window_params)
    tasks_std = ( (signal, window, is_ewm) for window in lookback_params)

    with mp.Pool(n_jobs) as pool:
        moving_average_results = pool.starmap(_moving_average_static, tasks_ma)
        moving_std_results = pool.starmap(_moving_std_static, tasks_std)

    moving_average_cache = {
        window: ma_window for window, ma_window in zip(window_params, moving_average_results)
    }

    moving_std_cache = {
        window: std_window for window, std_window in zip(lookback_params, moving_std_results)
    }

    new_signal = pd.concat({
        (f'{smooth}', f'{lookback}') : (
            ( moving_average_cache[smooth] - moving_average_cache[lookback] ) / moving_std_cache[lookback] 
        )
        for smooth in smooth_params
        for lookback in lookback_params
        if smooth * 2 <= lookback
    }, axis = 1).permute_levels(2)

    del moving_average_cache, moving_std_cache

    return new_signal.where(new_signal.abs() < 5, new_signal.apply(np.sign) * 5)


def cross_moving_average_(
        signal:pd.DataFrame, 
        smooth_params:Iterable[int] = (1, 2, 3, 4, 6, 8, 10,), 
        lookback_params:Iterable[int] = (2, 3, 4, 5, 6, 10, 12, 14, 17, 20, 28, 36, 44, 66,), 
        is_ewm:bool = False,
    ) -> pd.DataFrame: 
    """
    Compute the cross moving average of the signal df for different smooth and lookback parameters. 
    Augments the dimensionality of the signal df by two levels . 
    """

    @cache
    def moving_average(windows:int) -> pd.DataFrame:
        return (
            signal.apply(lambda x: x.dropna().rolling(windows).mean()) 
            if not is_ewm else 
            signal.apply(lambda x: x.dropna().ewm(halflife=windows).mean()) #ewm useless ? 
            )
    
    @cache
    def moving_std(windows:int) -> pd.DataFrame:
        return (
            signal.apply(lambda x: x.dropna().rolling(windows).std()) #.apply(lambda x: f(x.dropna())) to get rid of NaN
            if not is_ewm else 
            signal.apply(lambda x: x.dropna().ewm(halflife=windows).std()) 
            )
    
    new_signal = pd.concat({
        (f'{smooth}', f'{lookback}') : ( moving_average(smooth) - moving_average(lookback) ) / moving_std(lookback) 
        for smooth in smooth_params
        for lookback in lookback_params
        if smooth * 2 <= lookback
    }, axis = 1).permute_levels(2)
    
    moving_average.cache_clear()
    moving_std.cache_clear()

    return new_signal.where(new_signal.abs() < 5, new_signal.apply(np.sign) * 5)

def ranking(
    signal:pd.DataFrame, 
    k_values:Iterable[int] = (3, 5, 7,)
) -> pd.DataFrame: 
    """
    Compute the cross-sectional rank of the signal df. 
    Augments the dimensionality of the signal df by one level. 
    """
    def top_k_per_row(row:pd.Series, k:int) -> pd.Series:
        return row.isin(row.nlargest(k))

    def mask(row:pd.Series, k:int) -> pd.Series:
        return top_k_per_row(row, k).astype(int) - top_k_per_row(- row, k).astype(int)
        
    new_signal = pd.concat({
        f'{k}': signal.apply(lambda row: mask(row, k), axis = 1)
        for k in k_values
    }, axis = 1).permute_levels(1)

    return new_signal


def beta(
        ret: pd.Series, 
        factor: pd.DataFrame, 
        method: Literal['expanding', 'rolling', 'ewm'] = 'expanding', 
        lookback: int = 252
) -> pd.Series:     
    
    ret_, factor_ = ret.dropna(), factor.reindex(ret.dropna().index).ffill()
    
    methods = {
        'expanding': lambda x: x.expanding(),
        'rolling': lambda x: x.rolling(lookback),
        'ewm': lambda x: x.ewm(lookback)
    }
    
    if method not in methods:
        raise KeyError(f"method should be in {list(methods.keys())} not {method}")
        
    window = methods[method]
    return window(ret_).corr(factor_) / window(factor_).std() * window(ret_).std()

def infer(
    target: pd.DataFrame | pd.Series,
    features:pd.DataFrame, 
    model:object = ElasticNet(),
    train_every_n_steps:int = 30, 
    lookahead_steps:int = 0,
) -> pd.DataFrame:
  
    assert hasattr(model, 'fit') and hasattr(model, 'predict'), 'model should have fit and predict methods'
    assert isinstance(target, pd.DataFrame) or isinstance(target, pd.Series), 'target should be a pd.DataFrame or pd.Series'

    model_ts = ExpandingModel(model, train_every_n_steps, None, lookahead_steps)
    
    return model_ts.fit_predict(features, target.shift(lookahead_steps), skipna=True)

############ Markovitz

def markovitz_maxsharpe_static(pnl_train:pd.DataFrame, l2_reg:float = 0.5) -> pd.Series:
    #can have negative weights, useful for beta hedging / trading pairs & sectors

    try :
        n = len(pnl_train.columns) 
        mu = pnl_train.mean() 
        sigma = LedoitWolf().fit(pnl_train.dropna())._covariance + l2_reg * np.eye(n)
        weights = np.linalg.solve(sigma, mu)
    
    except ValueError as e: 
        print(f'error in markovitz_maxsharpe {e}')
        weights = 0

    return pd.Series(weights, index = pnl_train.columns).fillna(0)

_markovitz_maxsharpe_expanding = expanding_decorator(markovitz_maxsharpe_static)

def markovitz_maxsharpe(
        pnl_train:pd.DataFrame, 
        level:Literal['cross asset', 'per asset'] = 'per asset', 
        *args, **kwargs
    ) -> pd.DataFrame:  

    if level == 'cross asset':
        return _markovitz_maxsharpe_expanding(pnl_train, *args, **kwargs)
    elif level == 'per asset':
        return pd.concat({
        col: _markovitz_maxsharpe_expanding(pnl_train[col], *args, **kwargs)
        for col in pnl_train.columns.get_level_values(0).unique()
    }, axis = 1)
    else:
        raise ValueError(f"level should be in [0, -1] not {level}")


def _markovitz_minvol(pnl_train:pd.DataFrame, l2_reg:float) -> pd.Series: #   New
    #useful, barre's markovitz method
    #can't have negative weights, useful for isovol weighting of multiple signals
    return 


############ Clustering

def cluster(signal:pd.DataFrame) -> pd.DataFrame: #   New
    return 

##############################
##### Other stuff (useless ?)
##############################

def tgt(
    signal:pd.DataFrame, 
    smooth_params:Iterable[int] = (1, 2, 3, 4, 6, 8, 10,), 
    lookback_params:Iterable[int] = (2, 3, 4, 5, 6, 10, 12, 14, 17, 20, 28, 36, 44, 66,), 
    ) -> pd.DataFrame: 

    @cache
    def moving_average(windows:int) -> pd.DataFrame:
        return signal.rolling(windows).mean()

    new_signal = pd.concat({
        (f'{smooth}', f'{lookback}') : moving_average(smooth).pct_change(lookback) #variation: moving_average(smooth).diff(lookback) / lookback
        for smooth in smooth_params
        for lookback in lookback_params
        if smooth * 2 <= lookback
    }, axis = 1)
    
    moving_average.cache_clear()

    return new_signal.permute_levels(2)


def simple_buffer(
        signal:pd.DataFrame, 
        thresh:float = 1
        ) -> pd.DataFrame:

    return (
        signal
        .apply(np.sign)
        .where(signal.abs() > thresh, np.nan).ffill()
        )

def double_buffer(
    signal:pd.DataFrame, 
    thresh_top:float = 2, 
    thresh_bottom:float = 1
) -> pd.DataFrame:

    return (
        signal
        .apply(np.sign) 
        .where(signal.abs() > thresh_top, np.nan).ffill()
        .where(signal.abs() > thresh_bottom , 0)
    )

def macd(
    signal: pd.DataFrame,
    smooth_params:Iterable[int] = (1, 2, 3, 4, 6, 8, 10,), 
    lookback_params:Iterable[int] = (2, 3, 4, 5, 6, 10, 12, 14, 17, 20, 28, 36, 44, 66,), 
) -> pd.DataFrame:
        
    def _macd(signal_:pd.Series, smooth:int,lookback:int) -> pd.Series: 
        signal_ = signal_.dropna()
        cma = signal_.ewm(span=smooth, adjust=False).mean() - signal_.ewm(span=lookback, adjust=False).mean()
        cma_ma = cma.ewm(span=smooth, adjust=False).mean()

        bullish_crossover = (cma > cma_ma) & (cma.shift(1) <= cma_ma.shift(1))        
        bearish_crossover = (cma < cma_ma) & (cma.shift(1) >= cma_ma.shift(1)) 
        assert not (bullish_crossover & bearish_crossover).any()

        last_crossover_is_bullish = bullish_crossover.replace(False, np.nan).astype(int) # same as: pd.Series(1, index=signal_.index).where(bullish_crossover, np.nan)
        last_crossover_is_bearish = bearish_crossover.replace(False, np.nan).astype(int)
        
        return ( last_crossover_is_bullish - last_crossover_is_bearish ).ffill().fillna(0)

    return pd.concat({
        (f'{smooth}', f'{lookback}') : signal.apply(_macd, smooth = smooth, lookback = lookback)
        for smooth in smooth_params
        for lookback in lookback_params
        if smooth * 2 <= lookback
    }, axis = 1).permute_levels(2)

def balance(ds:pd.Series, n_bags:int = 3) -> pd.Series: #prends une série et renvoie pour chaque valeur dans quel quantile il se situe 
    result = pd.Series(0, index=ds.dropna().index)
    for k in range(1, n_bags):
        quantile = ds.dropna().expanding().quantile(k/n_bags)
        result += (ds.dropna() > quantile)
    
    return result - (n_bags - 1) / 2  