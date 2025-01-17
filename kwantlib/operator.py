import pandas as pd 
import numpy  as np 
import multiprocessing as mp 
from scipy.optimize import minimize

from typing import Iterable, List, Literal

from sklearn.covariance import LedoitWolf
from sklearn.linear_model import ElasticNet
from tskit_learn.timeseriesmodel import ExpandingModel

############ Operators

class Operator:

    n_jobs_value = mp.cpu_count() - 2

    @staticmethod
    def proj(signal:pd.DataFrame, threshold:float=0, level:int | List[int] = 0) -> pd.DataFrame: 
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
    def vote(signal:pd.DataFrame, threshold:float=0.7, level:int | List[int] = 0) -> pd.DataFrame: 
        """
        Reduce the dimensionality of the signal df by voting the different signals over the specified levels. 
        """
        return Operator.proj(signal.apply(np.sign), threshold = threshold, level = level)

    @staticmethod
    def _cross_moving_average_ds(
            signal:pd.Series, smooth_params:Iterable[int], lookback_params:Iterable[int], is_ewm:bool
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

        return new_signal.where(new_signal.abs() < 5, new_signal.apply(np.sign) * 5)

    @staticmethod
    def _cross_moving_average_df(
            signal:pd.DataFrame, smooth_params:Iterable[int], lookback_params:Iterable[int], 
            is_ewm:bool, skipna:bool, 
        ) -> pd.DataFrame:

        tasks = ( 
            (signal.loc[:, col].dropna() if skipna else signal.loc[:, col], smooth_params, lookback_params, is_ewm) 
            for col in signal.columns
        )
        with mp.Pool(Operator.n_jobs) as pool:
            results = pool.starmap(Operator._cross_moving_average_ds, tasks)
            
        return pd.concat({
            col: res for col, res in zip(signal.columns, results)
        }, axis=1)

    @staticmethod
    def cross_moving_average(        
            signal:pd.DataFrame | pd.Series, 
            smooth_params:Iterable[int] = (1, 2, 3, 4, 6, 8, 10,), 
            lookback_params:Iterable[int] = (2, 3, 4, 5, 6, 10, 12, 14, 17, 20, 28, 36, 44, 66,), 
            is_ewm:bool = False, skipna:bool = True,
        ) -> pd.DataFrame: 

        """
        Compute the cross moving average of the signal df for different smooth and lookback parameters. 
        Augments the dimensionality of the signal df by two levels . 
        """

        match type(signal):
            case pd.Series:
                return Operator._cross_moving_average_ds(signal, smooth_params, lookback_params, is_ewm)
            case pd.DataFrame:
                return Operator._cross_moving_average_df(signal, smooth_params, lookback_params, is_ewm, skipna)
            case _:
                raise ValueError(f"signal should be a pd.Series or pd.DataFrame not {type(signal)}")

    @staticmethod
    def _ranking_row(row:pd.Series, k:int) -> pd.DataFrame:
        top_k_per_row = row.isin(row.nlargest(k))
        bottom_k_per_row = row.isin(row.nsmallest(k))
        return top_k_per_row.astype(int) - bottom_k_per_row.astype(int)

    @staticmethod
    def _ranking_df(signal:pd.DataFrame, k:int) -> pd.DataFrame:
        tasks = ( (signal.loc[i, :].dropna(), k) for i in signal.index )
        with mp.Pool(Operator.n_jobs) as pool:
            results = pool.starmap(Operator._ranking_row, tasks)
        return pd.concat([res for res in results], axis = 1)

    @staticmethod
    def ranking(
            signal:pd.DataFrame, k_values:Iterable[int] = (3, 5, 7,),
        ) -> pd.DataFrame: 
        """
        Compute the cross-sectional rank of the signal df. 
        Augments the dimensionality of the signal df by one level. 
        """
        return pd.concat({
            f'{k}': Operator._ranking_df(signal.ffill(), k)
            for k in k_values
        }, axis = 1).reorder_levels(list(range(1, signal.columns.nlevels)) + [0])

    ############ Markovitz

    @staticmethod
    def _markovitz_minvol(pnl_train:pd.DataFrame, l2_reg:float) -> pd.Series:
        raise NotImplementedError('markovitz_minvol')
        try :
            n = len(pnl_train.columns) 
            mu = pnl_train.mean().to_numpy() 
            sigma = LedoitWolf().fit(pnl_train.dropna())._covariance + l2_reg * np.eye(n)
            def objective(beta:np.ndarray) -> np.ndarray:
                return - (beta @ mu) + beta @ sigma @ beta
            constraint =  ({'type': 'ineq', 'fun': lambda beta: np.sum(beta) - 1})
            bounds = [(1e-10, None) for _ in range(len(mu))]  # Contraintes pour que beta soit strictement positif
            initial_beta = np.ones(len(mu)) / len(mu)  # Initialisation des poids
            result = minimize(objective, initial_beta, bounds=bounds, constraints=constraint)
            return result.x
        except ValueError as e: 
            print(f'error in markovitz_minvol {e}')
            weights = 0
        return pd.Series(weights, index = pnl_train.columns).fillna(0)
    
    @staticmethod
    def _markovitz_maxsharpe(pnl_train:pd.DataFrame, l2_reg:float) -> pd.Series:
        try :
            n = len(pnl_train.columns) 
            mu = pnl_train.mean().to_numpy() 
            sigma = LedoitWolf().fit(pnl_train.dropna())._covariance + l2_reg * np.eye(n)
            weights = np.linalg.solve(sigma, mu)
        except ValueError as e: 
            print(f'error in markovitz_maxsharpe {e}')
            weights = 0
        return pd.Series(weights, index = pnl_train.columns).fillna(0)

    @staticmethod
    def _markovitz(
            pnl:pd.DataFrame, l2_reg:float, 
            method:Literal['maxsharpe', 'minvol'], 
            freq_retraining:int
        ) -> pd.DataFrame:
        training_dates = [pnl.index[i] for i in range(min(10, freq_retraining), len(pnl), freq_retraining)]

        match method:
            case 'maxsharpe':
                markovitz_func = Operator._markovitz_maxsharpe
            case 'minvol':
                markovitz_func = Operator._markovitz_minvol
            case _:
                raise ValueError(f"method should be in ['maxsharpe', 'minvol'] not {method}")

        tasks = ( (pnl.loc[ pnl.index < training_date, :], l2_reg) for training_date in training_dates )
        
        with mp.Pool(Operator.n_jobs) as pool:
            results = pool.starmap(markovitz_func, tasks)
        weights = pd.DataFrame(results, index = training_dates)
        
        return weights.reindex(pnl.index, method = 'ffill').fillna(0)
    
    @staticmethod   
    def markovitz(
            pnl:pd.DataFrame, l2_reg:float = 0.5, 
            level:Literal['cross asset', 'per asset'] = 'per asset', 
            method:Literal['maxsharpe', 'minvol'] = 'maxsharpe', 
            freq_retraining:int = 50, 
        ) -> pd.DataFrame:  
        """ 
        Markovitz weights trained in expanding window for a given frequency of retraining
        Maxsharpe: maximize the sharpe ratio with the possibility to go short
        Minvol: minimize the volatility of the portfolio with no possibility to go short
        """
        match level:
            case 'cross asset':
                return Operator._markovitz(pnl, l2_reg, method, freq_retraining)
            case 'per asset':
                assert isinstance(pnl.columns, pd.MultiIndex), 'pnl should have multiple levels to be train per asset'
                return pd.concat({
                    col: Operator._markovitz(pnl.loc[:, col].dropna(), l2_reg, method, freq_retraining)
                    for col in pnl.columns.get_level_values(0).unique()
                }, axis = 1)
            case _:
                raise ValueError(f"level should be in ['cross asset', 'per asset'] not {level}")

    ############ ML

    @staticmethod   
    def infer(
        target: pd.DataFrame | pd.Series,
        features:pd.DataFrame, 
        model:object = ElasticNet(),
        train_every_n_steps:int = 30, 
        lookahead_steps:int = 0,
    ) -> pd.DataFrame:
        """ 
        Infer the target from the features using a model trained in expanding window for a given frequency of retraining
        """
        assert hasattr(model, 'fit') and hasattr(model, 'predict'), 'model should have fit and predict methods'
        assert isinstance(target, pd.DataFrame) or isinstance(target, pd.Series), 'target should be a pd.DataFrame or pd.Series'
        model_ts = ExpandingModel(model, train_every_n_steps, None, lookahead_steps)
        return model_ts.fit_predict(features, target.shift(lookahead_steps), skipna=True)

    @staticmethod   
    def cluster(signal:pd.DataFrame) -> pd.DataFrame: 
        """ 
        Cluster the signal df using a clustering algorithm
        """
        raise NotImplementedError('cluster')
        return 
    
def monkey_patch_operator(): 
    pd.Series.cross_moving_average = Operator.cross_moving_average

    pd.DataFrame.cross_moving_average = Operator.cross_moving_average
    pd.DataFrame.proj = Operator.proj
    pd.DataFrame.vote = Operator.vote
    pd.DataFrame.ranking = Operator.ranking
    pd.DataFrame.markovitz = Operator.markovitz
    pd.DataFrame.infer = Operator.infer
    pd.DataFrame.cluster = Operator.cluster

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