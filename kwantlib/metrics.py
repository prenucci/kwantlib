import pandas as pd 
import numpy  as np 
import multiprocessing as mp 

from .utilitaires import Utilitaires

class Metrics:

    n_jobs = mp.cpu_count() - 2

    @staticmethod 
    def compute_position(signal:pd.DataFrame, volatility:pd.DataFrame) -> pd.DataFrame:
        signal = Utilitaires.custom_reindex_like(signal, volatility)
        pos = signal.div(volatility, axis = 0, level = 0) 
        pos = pos.where(Utilitaires.zscore(pos).abs() < 5, np.nan)
        return pos.ffill()
    
    @staticmethod
    def _compute_pnl_ds(position:pd.DataFrame, returns:pd.Series) -> pd.Series:
        assert returns.notna().all(), 'returns must not contain nan'
        pos = position.reindex(returns.index, method='ffill')
        return pos.shift(1).multiply(returns, axis=0)
    
    @staticmethod
    def _compute_pnl_df(position:pd.DataFrame, returns:pd.DataFrame) -> pd.DataFrame:
        
        tasks = (
            ( position.loc[:, [col]], returns.loc[:, col].dropna() ) 
            for col in returns.columns
        )
        with mp.Pool(Metrics.n_jobs) as pool:
            results = pool.map(Metrics._compute_pnl_ds, tasks)
        
        pnl = pd.concat(results, axis = 1)

        assert not pnl.apply(np.isinf).any().any(), 'inf in your pnl'

        return pnl
    
    @staticmethod
    def compute_pnl(position:pd.DataFrame, returns:pd.DataFrame | pd.Series) -> pd.DataFrame:
        match type(returns):
            case pd.Series:
                return Metrics._compute_pnl_ds(position, returns.dropna())
            case pd.DataFrame:
                return Metrics._compute_pnl_df(position, returns)
            case _:
                raise ValueError('returns must be a pd.DataFrame or pd.Series')
    
    @staticmethod
    def compute_returns(pnl:pd.DataFrame, pos:pd.DataFrame) -> pd.DataFrame:
        assert pos.columns.equals(pnl.columns), 'pos and pnl must have the same columns'
        return pnl.div(pos.abs().shift(1), axis=0)
    
    @staticmethod
    def compute_drawdown(pnl:pd.DataFrame) -> pd.Series:
        return - ( pnl.cumsum().cummax() - pnl.cumsum() ) / pnl.std()
    

    @staticmethod
    def compute_sharpe(pnl:pd.DataFrame | pd.Series) -> pd.Series | float:
        if hasattr(pnl.index, 'date'):
            pnl = pnl.groupby(pnl.index.date).sum()
        return 16 * pnl.mean() / pnl.std()
    
    @staticmethod
    def compute_ftrading(pos:pd.DataFrame | pd.Series) -> pd.Series | float:
        if hasattr(pos.index, 'date'):
            pos = pos.abs().groupby(pos.index.date).mean()
        return (pos > 0).mean()
    
    @staticmethod
    def compute_turnover(
        pos:pd.DataFrame | pd.Series, pos_change:pd.DataFrame | pd.Series = None
    ) -> pd.Series | float:
        pos_abs = pos.abs()
        if pos_change is None:
            pos_change = pos.diff().abs()
        if hasattr(pos.index, 'date'):
            pos_abs = pos_abs.groupby(pos_abs.index.date).mean()
            pos_change = pos_change.groupby(pos_change.index.date).sum()
        return 100 * pos_change.mean() / pos_abs.mean() 
    
    @staticmethod
    def compute_mean_returns(
        pnl:pd.DataFrame | pd.Series, pos:pd.DataFrame | pd.Series
    ) -> pd.Series | float:
        return pnl.mean() / pos.abs().mean()
    
    @staticmethod
    def compute_pnl_per_trade(
        pnl:pd.DataFrame | pd.Series, pos_change:pd.DataFrame | pd.Series
    ) -> pd.Series | float:
        if hasattr(pos_change.index, 'date'):
            pos_change = pos_change.groupby(pos_change.index.date).sum()
            pnl = pnl.groupby(pnl.index.date).sum()
        return 1e4 * pnl.mean() / pos_change.mean()
    
    @staticmethod
    def compute_maxdrawdown(pnl:pd.DataFrame | pd.Series) -> pd.Series | float:
        if hasattr(pnl.index, 'date'):
            pnl = pnl.groupby(pnl.index.date).sum()
        return ( pnl.cumsum().cummax() - pnl.cumsum() ).max() / pnl.std()
    
    @staticmethod
    def compute_calamar(pnl:pd.DataFrame | pd.Series) -> pd.Series | float:
        if hasattr(pnl.index, 'date'):
            pnl = pnl.groupby(pnl.index.date).sum()
        return 252 * pnl.mean() / Metrics.compute_maxdrawdown(pnl)
        return 16 * Metrics.compute_sharpe(pnl) / Metrics.compute_maxdrawdown(pnl)
    
    @staticmethod
    def compute_sortino(pnl:pd.DataFrame | pd.Series) -> pd.Series | float:
        if hasattr(pnl.index, 'date'):
            pnl = pnl.groupby(pnl.index.date).sum()
        return 16 * pnl.mean() / pnl[pnl < 0].std()
    
    @staticmethod
    def _compute_metrics_ds(pos:pd.Series, pnl:pd.Series, pos_change:pd.Series) -> pd.Series:
        return pd.Series({
            'eff_sharpe': Metrics.compute_sharpe(pnl[pnl!=0]),
            'raw_sharpe': Metrics.compute_sharpe(pnl),
            'turnover': Metrics.compute_turnover(pos, pos_change),
            'pnl_per_trade': Metrics.compute_pnl_per_trade(pnl, pos_change),
            'mean_returns': Metrics.compute_mean_returns(pnl, pos),
            'ftrading': Metrics.compute_ftrading(pos),
            'r_sharpe': Metrics.compute_sharpe(pnl.fillna(0).rolling(252).mean()),
            'maxdrawdown': Metrics.compute_maxdrawdown(pnl),
            'calamar': Metrics.compute_calamar(pnl),
            'sortino': Metrics.compute_sortino(pnl),
        }, name=pos.name)
    
    @staticmethod
    def _compute_metrics_df(pos:pd.DataFrame, pnl:pd.DataFrame, pos_change:pd.DataFrame) -> pd.DataFrame:
        tasks = (
            ( pos.loc[:, [col]], pnl.loc[:, col].fillna(0), pos_change.loc[:, col] ) 
            for col in pos.columns
        )
        with mp.Pool(Metrics.n_jobs) as pool:
            results = pool.map(Metrics._compute_metrics_ds, tasks)
        
        return (
            pd.concat(results, axis = 0)
            .sort_values(by='eff_sharpe', ascending=False, axis=0)
            .dropna(how='all', axis=0)
        )
    
    @staticmethod
    def compute_metrics(
        pos:pd.DataFrame | pd.Series, pnl:pd.DataFrame | pd.Series, pos_change:pd.DataFrame | pd.Series = None
    ) -> pd.DataFrame | pd.Series:
        
        if pos_change is None:
            pos_change = pos.diff().abs() 

        match (type(pos), type(pnl), type(pos_change)):
            case (pd.Series, pd.Series, pd.Series):
                return Metrics._compute_metrics_ds(pos, pnl, pos_change)
            case (pd.DataFrame, pd.DataFrame, pd.DataFrame):
                assert pos.columns.equals(pnl.columns) and pos.columns.equals(pos_change.columns), 'pos, pnl and pos_change must have the same columns'
                return Metrics._compute_metrics_df(pos, pnl, pos_change)
            case _:
                raise ValueError('pos, pnl and pos_change must be of the same type')
        

    @staticmethod
    def backtest(pos:pd.DataFrame, pnl:pd.DataFrame, pos_change:pd.DataFrame = None, risk:float = 1) -> pd.DataFrame:   
        if pos_change is None:
            pos_change = pos.diff().abs()

        if hasattr(pos.index, 'date'):
            pos = pos.groupby(pos.index.date).mean()
            pos_change = pos_change.groupby(pos_change.index.date).sum()
            pnl = pnl.groupby(pnl.index.date).sum()

        print(Metrics.compute_metrics(pos.abs().sum(1), pnl.sum(1), pos_change.sum(1)))

        Utilitaires.plotx( risk * pnl.sum(1).cumsum() / pnl.sum(1).std(), title='pnl total' ).show()
        Utilitaires.plotx( risk * Metrics.compute_drawdown(pnl.sum(1)), title='drawdown' ).show()

        if len(pnl.columns) < 30:
            Utilitaires.plotx( risk * pnl.cumsum() / pnl.std(), title='pnl per asset' ).show()

        return Metrics.compute_metrics(pos, pnl, pos_change)