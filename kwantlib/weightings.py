import pandas as pd 
import numpy as np 
import multiprocessing as mp 

from typing import Literal
from scipy.optimize import minimize

from .utilitaires import Utilitaires

class Weighting:

    @staticmethod
    def _markovitz_minvol(pnl_train:pd.DataFrame, l2_reg:float) -> pd.Series:
        try:
            n = len(pnl_train.columns) 
            sigma = pnl_train.cov().to_numpy() 
            result = minimize(
                fun= lambda beta: beta @ sigma @ beta + l2_reg * beta @ beta, 
                x0= np.linalg.solve(sigma, np.ones(n)), 
                bounds= [(0, None) for _ in range(n)], 
                constraints= [{'type': 'eq', 'fun': lambda beta: beta @ beta - 1} ], 
                method= 'SLSQP'
            )
            if not result.success:
                raise ValueError(f"Optimization failed: {result.message}") 
            weights = result.x
            weights /= np.sqrt(weights @ weights)           
        except ValueError as e: 
            print(f'error in markovitz_minvol {e}, filling the corresponding row with nan')
            weights = np.nan
        return pd.Series(weights, index = pnl_train.columns)


    @staticmethod
    def _markovitz_maxsharpe(pnl_train:pd.DataFrame, l2_reg:float) -> pd.Series:
        assert not pnl_train.dropna().empty, 'pnl_train is empty'
        try :
            n = len(pnl_train.columns) 
            mu = pnl_train.mean().to_numpy() 
            sigma = pnl_train.cov().to_numpy() + l2_reg * np.eye(n)
            weights = np.linalg.solve(sigma, mu) 
            weights /= np.sqrt(weights @ weights)
        except ValueError as e: 
            print(f'error in markovitz_maxsharpe {e}, filling the corresponding row with nan')
            weights = np.nan
        return pd.Series(weights, index = pnl_train.columns)

    @staticmethod
    def _markovitz(
            pnl:pd.DataFrame, l2_reg:float, 
            method:Literal['maxsharpe', 'minvol'], 
            freq_retraining:int
        ) -> pd.DataFrame:
        
        match method:
            case 'maxsharpe':
                markovitz_func = Weighting._markovitz_maxsharpe
            case 'minvol':
                markovitz_func = Weighting._markovitz_minvol
            case _:
                raise ValueError(f"method should be in ['maxsharpe', 'minvol'] not {method}")
            
        dates = pnl.sort_index(axis=0).index.unique()
        training_dates = (dates[i] for i in range(min(10, freq_retraining), len(dates), freq_retraining))

        tasks = ( 
            (pnl.loc[ pnl.index < training_date, :].fillna(0), l2_reg) for training_date in training_dates
        )

        with mp.Pool(Utilitaires.n_jobs) as pool:
            results = pool.starmap(markovitz_func, tasks)
        weights = pd.DataFrame(results, index = training_dates)
        
        return weights.reindex(pnl.index, method = 'ffill').ffill().fillna(0)

    @staticmethod   
    def markovitz(
            pnl:pd.DataFrame, l2_reg:float = 0.5, 
            level:Literal['cross asset', 'per asset'] = 'cross asset', 
            method:Literal['maxsharpe', 'minvol'] = 'maxsharpe', 
            freq_retraining:int = 50, 
        ) -> pd.DataFrame:  
        """ 
        Markovitz weights trained in expanding window for a given frequency of retraining
        Maxsharpe: maximize the sharpe ratio with the possibility to go short
        Minvol: minimize the volatility of the portfolio with no possibility to go short
        """
        if hasattr(pnl.index, "date"):
            pnl = pnl.fillna(0).groupby(pnl.index.date).sum()

        match level:
            case 'cross asset':
                return Weighting._markovitz(pnl, l2_reg, method, freq_retraining)
            case 'per asset':
                assert isinstance(pnl.columns, pd.MultiIndex), 'pnl should have multiple levels to be train per asset'
                return pd.concat({
                    col: Weighting._markovitz(pnl.loc[:, col].dropna(), l2_reg, method, freq_retraining)
                    for col in pnl.columns.get_level_values(0).unique()
                }, axis = 1)
            case _:
                raise ValueError(f"level should be in ['cross asset', 'per asset'] not {level}")
            
    @staticmethod
    def monkey_patch(): 
        pd.DataFrame.markovitz = Weighting.markovitz