import pandas as pd 
import multiprocessing as mp 

from .utilitaires import Utilitaires
from .core import Core

class Metrics:
        
    @staticmethod
    def sharpe(pnl:pd.DataFrame | pd.Series) -> pd.Series | float:
        return 16 * pnl.mean() / pnl.std()
    
    @staticmethod
    def turnover(
        pos:pd.DataFrame | pd.Series, pos_change:pd.DataFrame | pd.Series
    ) -> pd.Series | float:
        return 100 * pos_change.mean() / pos.abs().mean() 
    
    @staticmethod
    def pnl_per_trade(
        pnl:pd.DataFrame | pd.Series, pos_change:pd.DataFrame | pd.Series
    ) -> pd.Series | float:
        return 1e4 * pnl.mean() / pos_change.mean()
    
    @staticmethod
    def mean_returns(
        pos:pd.DataFrame | pd.Series, pnl:pd.DataFrame | pd.Series,
    ) -> pd.Series | float:
        return pnl.mean() / pos.abs().mean()
    
    @staticmethod
    def maxdrawdown(pnl:pd.DataFrame | pd.Series) -> pd.Series | float:
        return - Core.compute_drawdown(pnl).min() / pnl.std()
    
    @staticmethod
    def calamar(pnl:pd.DataFrame | pd.Series) -> pd.Series | float:
        return 16 * Metrics.sharpe(pnl) / Metrics.maxdrawdown(pnl)
    
    @staticmethod
    def sortino(pnl:pd.DataFrame | pd.Series) -> pd.Series | float:
        return 16 * pnl.mean() / pnl[pnl < 0].std()
    
    @staticmethod
    def ftrading(pos:pd.DataFrame | pd.Series) -> pd.Series | float:
        return (pos.abs() != 0).mean()
    
    @staticmethod
    def win_rate(pnl:pd.DataFrame | pd.Series) -> pd.Series | float:
        return (pnl > 0).sum() / ( (pnl != 0).sum() )
    
    @staticmethod
    def long_ratio(pnl:pd.DataFrame | pd.Series, pos:pd.DataFrame | pd.Series) -> pd.Series | float:
        pnl = pnl.fillna(0)
        pos_shifted = pos.shift(1).reindex(pnl.index, method='ffill').ffill()
        return pnl.where(pos_shifted > 0, 0).sum() / pnl.sum()
    
    ### Backtest ###

    @staticmethod
    def _resample_daily(
        pos_abs:pd.DataFrame | pd.Series, pnl:pd.DataFrame | pd.Series, pos_change:pd.DataFrame | pd.Series
    ) -> tuple[pd.DataFrame | pd.Series, pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]:
        if hasattr(pos_abs.index, 'date'):
            pos_abs = pos_abs.groupby(pos_abs.index.date).sum()
        if hasattr(pnl.index, 'date'):
            pnl = pnl.groupby(pnl.index.date).sum()
        if hasattr(pos_change.index, 'date'):
            pos_change = pos_change.groupby(pos_change.index.date).sum()
        return pos_abs, pnl, pos_change
    
    @staticmethod
    def _metrics_ds(pos:pd.Series, pnl:pd.Series, pos_change:pd.Series) -> pd.Series:
        return pd.Series({
            'eff_sharpe': Metrics.sharpe(pnl[pnl!=0]),
            'raw_sharpe': Metrics.sharpe(pnl),
            'turnover': Metrics.turnover(pos, pos_change),
            'pnl_per_trade': Metrics.pnl_per_trade(pnl, pos_change),
            'mean_returns': Metrics.mean_returns(pnl, pos),
            'maxdrawdown': Metrics.maxdrawdown(pnl),
            'calamar': Metrics.calamar(pnl),
            'sortino': Metrics.sortino(pnl),
            'ftrading': Metrics.ftrading(pos),
            'win_rate': Metrics.win_rate(pnl),
            # 'long_ratio': Metrics.long_ratio(pnl, pos),
            'r_sharpe': Metrics.sharpe(pnl.fillna(0).rolling(252).mean()),
        })
    
    @staticmethod
    def _metrics_df(pos:pd.DataFrame, pnl:pd.DataFrame, pos_change:pd.DataFrame) -> pd.DataFrame:
        tasks = (
            ( pos.loc[:, col], pnl.loc[:, col].fillna(0), pos_change.loc[:, col] ) 
            for col in pos.columns
        )
        with mp.Pool(Utilitaires.n_jobs) as pool:
            results = pool.starmap(Metrics._metrics_ds, tasks)
        
        return pd.concat({
            col:result_col for col, result_col in zip(pos.columns, results)
        }, axis=1).T.sort_values(by='eff_sharpe', ascending=False)
        
    @staticmethod
    def metrics(
        pos:pd.DataFrame | pd.Series, pnl:pd.DataFrame | pd.Series, pos_change:pd.DataFrame | pd.Series = None
    ) -> pd.DataFrame | pd.Series:
        
        if pos_change is None:
            pos_change = pos.diff().abs() 

        pos_abs, pnl, pos_change = Metrics._resample_daily(pos.abs(), pnl, pos_change)

        match (type(pos_abs), type(pnl), type(pos_change)):
            case (pd.Series, pd.Series, pd.Series):
                return Metrics._metrics_ds(pos_abs, pnl, pos_change)
            case (pd.DataFrame, pd.DataFrame, pd.DataFrame):
                assert pos_abs.columns.equals(pnl.columns) and pos_abs.columns.equals(pos_change.columns), 'pos, pnl and pos_change must have the same columns'
                return Metrics._metrics_df(pos_abs, pnl, pos_change)
            case _:
                raise ValueError('pos, pnl and pos_change must be of the same type')
        
    @staticmethod
    def backtest(
        pos:pd.DataFrame, pnl:pd.DataFrame, pos_change:pd.DataFrame = None, risk:float = 1
    ) -> pd.DataFrame:   
        
        if pos_change is None:
            pos_change = pos.diff().abs()

        pos_abs, pnl, pos_change = Metrics._resample_daily(pos.abs(), pnl, pos_change)

        print( Metrics.metrics(pos_abs.sum(1), pnl.sum(1), pos_change.sum(1)).to_frame('overall').T )

        Utilitaires.plotx( risk * pnl.sum(1).cumsum() / pnl.sum(1).std(), title= 'pnl total' ).show()
        Utilitaires.plotx( risk * Core.compute_drawdown(pnl.sum(1)) / pnl.sum(1).std(), title='drawdown' ).show()

        if len(pnl.columns) < 30:
            Utilitaires.plotx( risk * pnl.cumsum() / pnl.std(), title='pnl per asset' ).show()

        return Metrics.metrics(pos_abs, pnl, pos_change)
    
    @staticmethod
    def quick_backtest(
        returns:pd.DataFrame, signal:pd.DataFrame, 
        vol:pd.DataFrame = None, bid_ask_spread:pd.DataFrame = None, 
        risk:float = 1, fee_per_transaction:float = 1e-4
    ) -> pd.DataFrame:
        
        pos = Core.compute_position(signal, vol) if vol is not None else signal
        pnl = Core.compute_pnl(pos, returns)
        if bid_ask_spread is not None:
            pnl -= Core.compute_cost(pos.diff().abs(), bid_ask_spread, fee_per_transaction)
        return Metrics.backtest(pos, pnl, pos.diff().abs(), risk)
