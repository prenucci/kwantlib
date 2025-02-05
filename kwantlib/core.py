import pandas as pd 
import numpy as np 
import multiprocessing as mp 
from typing import Iterable
from .utilitaires import Utilitaires

class Core:

    @staticmethod 
    def compute_position(signal:pd.DataFrame, volatility:pd.DataFrame) -> pd.DataFrame:
        signal = Utilitaires.custom_reindex_like(signal, volatility)
        pos = signal.div(volatility, axis = 0, level = 0) 
        pos = pos.where(Utilitaires.zscore(pos).abs() < 5, np.nan)
        return pos.ffill().fillna(0)
    
    @staticmethod
    def _compute_pnl_ds(position:pd.DataFrame | pd.Series, returns:pd.Series) -> pd.DataFrame | pd.Series:
        assert returns.notna().all(), 'returns must not contain nan'
        pos = position.reindex(returns.index, method='ffill')
        return pos.shift(1).multiply(returns, axis=0)
    
    @staticmethod
    def _compute_pnl_df(position:pd.DataFrame, returns:pd.DataFrame) -> pd.DataFrame:
        tasks = ( ( position.loc[:, [col]], returns.loc[:, col].dropna() ) for col in returns.columns )
        with mp.Pool(Utilitaires.n_jobs) as pool:
            results = pool.starmap(Core._compute_pnl_ds, tasks)
        pnl = pd.concat(results, axis = 1)
        assert not pnl.apply(np.isinf).any().any(), 'inf in your pnl'
        return pnl
    
    @staticmethod
    def compute_pnl(position:pd.DataFrame | pd.Series, returns:pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        match (type(position), type(returns)):
            case (pd.Series, pd.Series):
                return Core._compute_pnl_ds(position, returns.dropna())
            case (pd.DataFrame, pd.DataFrame | pd.Series):
                return Core._compute_pnl_df(position, returns)
            case _:
                raise ValueError('returns must be a pd.DataFrame or pd.Series')
            
    @staticmethod
    def _compute_cost_ds(
        pos_change:pd.DataFrame | pd.Series, bid_ask_spread:pd.Series, fee_per_transaction:float
    ) -> pd.DataFrame | pd.Series:
        assert bid_ask_spread.notna().all(), 'returns must not contain nan'
        pos_change = pos_change.reindex(bid_ask_spread.index, method='ffill').ffill()
        return (bid_ask_spread / 2 + fee_per_transaction) * pos_change 
    
    @staticmethod
    def _compute_cost_df(
        pos_change:pd.DataFrame, bid_ask_spread:pd.DataFrame, fee_per_transaction:float
    ) -> pd.DataFrame:
        tasks = (
            ( pos_change.loc[:, [col]], bid_ask_spread.loc[:, col].dropna(), fee_per_transaction ) 
            for col in pos_change.columns
        )
        with mp.Pool(Utilitaires.n_jobs) as pool:
            results = pool.starmap(Core._compute_cost_ds, tasks)
        return pd.concat(results, axis = 1)

    @staticmethod
    def compute_cost(
        pos_change:pd.DataFrame | pd.Series, bid_ask_spread:pd.DataFrame | pd.Series, 
        fee_per_transaction:float
    ) -> pd.DataFrame | pd.Series:
        match (type(pos_change), type(bid_ask_spread)):
            case (pd.Series, pd.Series):
                return Core._compute_cost_ds(pos_change, bid_ask_spread, fee_per_transaction)
            case (pd.DataFrame, pd.DataFrame | pd.Series):
                return Core._compute_cost_df(pos_change, bid_ask_spread, fee_per_transaction)
            case _:
                raise ValueError(
                    f'(pd.Series, pd.Series) or (pd.DataFrame, pd.DataFrame | pd.Series) are the only valid types. Not {type(pos_change), type(bid_ask_spread)}'
                )

    @staticmethod
    def _compute_ret_ds(pos:pd.Series, pnl:pd.Series) -> pd.Series: 
        pos = pos.reindex(pnl.index, method='ffill').ffill().shift(1).fillna(0)
        return pnl.div(pos.abs()).fillna(0)
    
    @staticmethod
    def _compute_ret_df(pos:pd.DataFrame, pnl:pd.DataFrame) -> pd.DataFrame:
        tasks = ( (pos.loc[:, [col]], pnl.loc[:, col].dropna()) for col in pos.columns )
        with mp.Pool(Utilitaires.n_jobs) as pool:
            results = pool.starmap(Core._compute_ret_ds, tasks)
        return pd.concat(results, axis = 1)
    
    @staticmethod
    def compute_ret(pos:pd.DataFrame | pd.Series, pnl:pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        match (type(pos), type(pnl)):
            case (pd.Series, pd.Series):
                return Core._compute_ret_ds(pos, pnl)
            case (pd.DataFrame, pd.DataFrame | pd.Series):
                return Core._compute_ret_df(pos, pnl)
            case _:
                raise ValueError(f'pos and pnl must be a pd.Series or pd.DataFrame not {type(pos), type(pnl)}')
    
    @staticmethod
    def compute_compounded_value(pos:pd.DataFrame, pnl:pd.DataFrame) -> pd.DataFrame:
        return (1 + Core.compute_ret(pos, pnl)).cumprod()
    
    @staticmethod
    def compute_drawdown(pnl:pd.DataFrame) -> pd.Series:
        return - ( pnl.cumsum().cummax() - pnl.cumsum() ) 
    
