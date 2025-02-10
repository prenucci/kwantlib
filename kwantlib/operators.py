import pandas as pd 
import numpy  as np 
import multiprocessing as mp 
from typing import Iterable, Callable, Any, Tuple, Dict, Literal

from .utilitaires import Utilitaires

class Operator:

    @staticmethod
    def _zscore_ds(
        ds:pd.Series, method:Literal['expanding', 'rolling', 'ewm'] = 'expanding', lookback:int = 252
    ) -> pd.Series:
        match method:
            case 'expanding':
                return (ds - ds.expanding().mean()) / ds.expanding().std()
            case 'rolling':
                return (ds - ds.rolling(lookback).mean()) / ds.rolling(lookback).std()
            case 'ewm':
                return (ds - ds.ewm(lookback).mean()) / ds.ewm(lookback).std()
            case _:
                raise ValueError(f"method should be in ['expanding', 'rolling', 'ewm'] not {method}")
            
    @staticmethod
    def zscore(
        df: pd.DataFrame | pd.Series, 
        method: Literal['expanding', 'rolling', 'ewm'] = 'expanding', 
        lookback: int = 252, skipna: bool = True
    ) -> pd.DataFrame | pd.Series: 
        
        match type(df):
            case pd.Series:
                zscore = Operator._zscore_ds(df.dropna() if skipna else df, method, lookback)
            case pd.DataFrame:
                zscore = df.apply(lambda x: Operator._zscore_ds(x.dropna() if skipna else x, method, lookback))
            case _:
                raise ValueError(f"df should be a pd.Series or pd.DataFrame not {type(df)}")
        
        return zscore.reindex(df.index).ffill()

    @staticmethod
    def proj(signal:pd.DataFrame, threshold:float=0, level:int | Iterable[int] = 0) -> pd.DataFrame: 
        """
        Reduce the dimensionality of the signal df by averaging over the specified levels. 
        """
        match level:
            case -1:
                new_signal = signal.mean(axis = 1).to_frame('projected')
            case _:
                new_signal = signal.groupby(level = level, axis = 1).mean()
        return new_signal.where(new_signal.abs() >= threshold, 0)
    
    @staticmethod
    def vote(signal:pd.DataFrame, threshold:float=0.7, level:int | Iterable[int] = 0) -> pd.DataFrame: 
        """
        Reduce the dimensionality of the signal df by voting the different signals over the specified levels. 
        """
        return Operator.proj(signal.apply(np.sign), threshold = threshold, level = level)

    @staticmethod
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

        return new_signal if not is_proj else new_signal.mean(1)
    
    @staticmethod
    def _cross_moving_average_df(
            signal:pd.DataFrame, 
            smooth_params:Iterable[int], 
            lookback_params:Iterable[int], 
            is_proj:bool, is_ewm:bool, skipna:bool, 
        ) -> pd.DataFrame:

        tasks = ( 
            (
                signal.loc[:, col].dropna() if skipna else signal.loc[:, col], 
                smooth_params, lookback_params, is_proj, is_ewm
            ) 
            for col in signal.columns
        )
        with mp.Pool(Utilitaires.n_jobs) as pool:
            results = pool.starmap(Operator._cross_moving_average_ds, tasks)
            
        return pd.concat({
            col: res for col, res in zip(signal.columns, results)
        }, axis=1)
    
    @staticmethod
    def cross_moving_average(        
            signal:pd.DataFrame | pd.Series, 
            smooth_params:Iterable[int] = (1, 2, 3, 4, 6, 8, 10,), 
            lookback_params:Iterable[int] = (2, 3, 4, 5, 6, 10, 12, 14, 17, 20, 28, 36, 44, 66,), 
            is_proj:bool = False, is_ewm:bool = False, skipna:bool = True, 
        ) -> pd.DataFrame: 

        """
        Compute the cross moving average of the signal df for different smooth and lookback parameters. 
        Augments the dimensionality of the signal df by two levels . 
        """

        match type(signal):
            case pd.Series:
                return Operator._cross_moving_average_ds(
                    signal, smooth_params, lookback_params, is_proj, is_ewm
                )
            case pd.DataFrame:
                return Operator._cross_moving_average_df(
                    signal, smooth_params, lookback_params, is_proj, is_ewm, skipna, 
                )
            case _:
                raise ValueError(f"signal should be a pd.Series or pd.DataFrame not {type(signal)}")

    @staticmethod
    def _is_top_k(row:pd.Series, k:int) -> pd.DataFrame:
        threshold = row.nlargest(k).iloc[-1]
        return (row > threshold).astype(int)

    @staticmethod
    def _ranking_row(row:pd.Series, k:int) -> pd.DataFrame:
        return Operator._is_top_k(row, k) - Operator._is_top_k(-row, k)

    @staticmethod
    def _ranking_df(signal:pd.DataFrame, k:int) -> pd.DataFrame:
        tasks = ( (signal.loc[i, :].dropna(), k) for i in signal.index )
        with mp.Pool(Utilitaires.n_jobs) as pool:
            results = pool.starmap(Operator._ranking_row, tasks)
        return pd.DataFrame([res for res in results], columns = signal.columns)

    @staticmethod
    def ranking(
            signal:pd.DataFrame, k_values:Iterable[int] = (3, 5, 7,),
        ) -> pd.DataFrame: 
        """
        Compute the cross-sectional rank of the signal df. 
        Augments the dimensionality of the signal df by one level. 
        """
        new_signal = pd.concat({
            f'{k}': Operator._ranking_df(signal.ffill(), k)
            for k in k_values
        }, axis = 1)
        
        new_signal = new_signal.reorder_levels(
            list(range(1, signal.columns.nlevels+1)) + [0], 
        axis=1).sort_index(axis=1)

        return new_signal
    
    @staticmethod
    def monkey_patch(): 
        pd.Series.zscore = Operator.zscore
        pd.DataFrame.zscore = Operator.zscore
        pd.Series.cross_moving_average = Operator.cross_moving_average
        pd.DataFrame.cross_moving_average = Operator.cross_moving_average
        pd.DataFrame.proj = Operator.proj
        pd.DataFrame.vote = Operator.vote
        pd.DataFrame.ranking = Operator.ranking

    @staticmethod
    def chain(
        signal:pd.DataFrame, 
        operators:Iterable[Tuple[Callable[[pd.DataFrame, Any], pd.DataFrame], Tuple[Any], Dict[str, Any]]]
    ) -> pd.DataFrame:
        for operator, *rest in operators:
            args, kwargs = (), {}
            for item in rest:
                if isinstance(item, tuple):
                    args = item
                elif isinstance(item, dict):
                    kwargs = item
                else:
                    raise ValueError(f"Invalid item type: {type(item)}")
            signal = operator(signal, *args, **kwargs)
        return signal

    # ##############################
    # ##### Other stuff (useless ?)
    # ##############################

    # def tgt(
    #     signal:pd.DataFrame, 
    #     smooth_params:Iterable[int] = (1, 2, 3, 4, 6, 8, 10,), 
    #     lookback_params:Iterable[int] = (2, 3, 4, 5, 6, 10, 12, 14, 17, 20, 28, 36, 44, 66,), 
    #     ) -> pd.DataFrame: 

    #     @cache
    #     def moving_average(windows:int) -> pd.DataFrame:
    #         return signal.rolling(windows).mean()

    #     new_signal = pd.concat({
    #         (f'{smooth}', f'{lookback}') : moving_average(smooth).pct_change(lookback) #variation: moving_average(smooth).diff(lookback) / lookback
    #         for smooth in smooth_params
    #         for lookback in lookback_params
    #         if smooth * 2 <= lookback
    #     }, axis = 1)
        
    #     moving_average.cache_clear()

    #     return new_signal.permute_levels(2)


    # def simple_buffer(
    #         signal:pd.DataFrame, 
    #         thresh:float = 1
    #         ) -> pd.DataFrame:

    #     return (
    #         signal
    #         .apply(np.sign)
    #         .where(signal.abs() > thresh, np.nan).ffill()
    #         )

    # def double_buffer(
    #     signal:pd.DataFrame, 
    #     thresh_top:float = 2, 
    #     thresh_bottom:float = 1
    # ) -> pd.DataFrame:

    #     return (
    #         signal
    #         .apply(np.sign) 
    #         .where(signal.abs() > thresh_top, np.nan).ffill()
    #         .where(signal.abs() > thresh_bottom , 0)
    #     )

    # def macd(
    #     signal: pd.DataFrame,
    #     smooth_params:Iterable[int] = (1, 2, 3, 4, 6, 8, 10,), 
    #     lookback_params:Iterable[int] = (2, 3, 4, 5, 6, 10, 12, 14, 17, 20, 28, 36, 44, 66,), 
    # ) -> pd.DataFrame:
            
    #     def _macd(signal_:pd.Series, smooth:int,lookback:int) -> pd.Series: 
    #         signal_ = signal_.dropna()
    #         cma = signal_.ewm(span=smooth, adjust=False).mean() - signal_.ewm(span=lookback, adjust=False).mean()
    #         cma_ma = cma.ewm(span=smooth, adjust=False).mean()

    #         bullish_crossover = (cma > cma_ma) & (cma.shift(1) <= cma_ma.shift(1))        
    #         bearish_crossover = (cma < cma_ma) & (cma.shift(1) >= cma_ma.shift(1)) 
    #         assert not (bullish_crossover & bearish_crossover).any()

    #         last_crossover_is_bullish = bullish_crossover.replace(False, np.nan).astype(int) # same as: pd.Series(1, index=signal_.index).where(bullish_crossover, np.nan)
    #         last_crossover_is_bearish = bearish_crossover.replace(False, np.nan).astype(int)
            
    #         return ( last_crossover_is_bullish - last_crossover_is_bearish ).ffill().fillna(0)

    #     return pd.concat({
    #         (f'{smooth}', f'{lookback}') : signal.apply(_macd, smooth = smooth, lookback = lookback)
    #         for smooth in smooth_params
    #         for lookback in lookback_params
    #         if smooth * 2 <= lookback
    #     }, axis = 1).permute_levels(2)

    # def balance(ds:pd.Series, n_bags:int = 3) -> pd.Series: #prends une série et renvoie pour chaque valeur dans quel quantile il se situe 
    #     result = pd.Series(0, index=ds.dropna().index)
    #     for k in range(1, n_bags):
    #         quantile = ds.dropna().expanding().quantile(k/n_bags)
    #         result += (ds.dropna() > quantile)
        
    #     return result - (n_bags - 1) / 2              