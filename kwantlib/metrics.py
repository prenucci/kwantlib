import pandas as pd 
import numpy  as np 
import multiprocessing as mp 

from .utilitaires import Utilitaires

class Metrics:

    n_jobs = mp.cpu_count() - 2

    ### Core functions ###  

    @staticmethod 
    def compute_position(signal:pd.DataFrame, volatility:pd.DataFrame) -> pd.DataFrame:
        signal = Utilitaires.custom_reindex_like(signal, volatility)
        pos = signal.div(volatility, axis = 0, level = 0) 
        pos = pos.where(Utilitaires.zscore(pos).abs() < 5, np.nan)
        return pos.ffill()
    
    @staticmethod
    def _compute_pnl_ds(position:pd.DataFrame | pd.Series, returns:pd.Series) -> pd.DataFrame | pd.Series:
        assert returns.notna().all(), 'returns must not contain nan'
        pos = position.reindex(returns.index, method='ffill')
        return pos.shift(1).multiply(returns, axis=0)
    
    @staticmethod
    def _compute_pnl_df(position:pd.DataFrame, returns:pd.DataFrame) -> pd.DataFrame:
        tasks = ( ( position.loc[:, [col]], returns.loc[:, col].dropna() ) for col in returns.columns )
        with mp.Pool(Metrics.n_jobs) as pool:
            results = pool.starmap(Metrics._compute_pnl_ds, tasks)
        pnl = pd.concat(results, axis = 1)
        assert not pnl.apply(np.isinf).any().any(), 'inf in your pnl'
        return pnl
    
    @staticmethod
    def compute_pnl(position:pd.DataFrame | pd.Series, returns:pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        match (type(position), type(returns)):
            case (pd.Series, pd.Series):
                return Metrics._compute_pnl_ds(position, returns.dropna())
            case (pd.DataFrame, pd.DataFrame | pd.Series):
                return Metrics._compute_pnl_df(position, returns)
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
        with mp.Pool(Metrics.n_jobs) as pool:
            results = pool.starmap(Metrics._compute_cost_ds, tasks)
        return pd.concat(results, axis = 1)

    @staticmethod
    def compute_cost(
        pos_change:pd.DataFrame | pd.Series, bid_ask_spread:pd.DataFrame | pd.Series, fee_per_transaction:float
    ) -> pd.DataFrame | pd.Series:
        match (type(pos_change), type(bid_ask_spread)):
            case (pd.Series, pd.Series):
                return Metrics._compute_cost_ds(pos_change, bid_ask_spread, fee_per_transaction)
            case (pd.DataFrame, pd.DataFrame | pd.Series):
                return Metrics._compute_cost_df(pos_change, bid_ask_spread, fee_per_transaction)
            case _:
                raise ValueError(
                    f'(pd.Series, pd.Series) or (pd.DataFrame, pd.DataFrame | pd.Series) are the only valid types. Not {type(pos_change), type(bid_ask_spread)}'
                )
                
    @staticmethod
    def compute_ret(pos:pd.DataFrame, pnl:pd.DataFrame) -> pd.DataFrame:
        assert pos.columns.equals(pnl.columns), 'pos and pnl must have the same columns'
        return pnl.div(
            pos.abs().shift(1), axis=0
        )
    
    @staticmethod
    def compute_compounded_value(pos:pd.DataFrame, pnl:pd.DataFrame) -> pd.DataFrame:
        return_pnl = Metrics.compute_ret(pos, pnl)
        return (1 + return_pnl).cumprod()
    
    @staticmethod
    def compute_drawdown(pnl:pd.DataFrame) -> pd.Series:
        return - ( pnl.cumsum().cummax() - pnl.cumsum() ) 
    

    ### Metrics ###

    @staticmethod
    def _resample_daily(
        pos_abs: pd.DataFrame | pd.Series = None, 
        pnl: pd.DataFrame | pd.Series = None, 
        pos_change:pd.DataFrame | pd.Series = None
    ) -> tuple[pd.DataFrame | pd.Series, pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]:
        if pos_abs is not None and hasattr(pos_abs.index, 'date'):
            pos_abs = pos_abs.groupby(pos_abs.index.date).mean()
        if pnl is not None and hasattr(pnl.index, 'date'):
            pnl = pnl.groupby(pnl.index.date).sum()
        if pos_change is not None and hasattr(pos_change.index, 'date'):
            pos_change = pos_change.groupby(pos_change.index.date).sum()
        return pos_abs, pnl, pos_change
    
    @staticmethod
    def compute_sharpe(pnl:pd.DataFrame | pd.Series) -> pd.Series | float:
        _, pnl, _ = Metrics._resample_daily(None, pnl, None)
        return 16 * pnl.mean() / pnl.std()
    
    @staticmethod
    def compute_turnover(
        pos:pd.DataFrame | pd.Series, pos_change:pd.DataFrame | pd.Series
    ) -> pd.Series | float:
        pos_abs, _, pos_change = Metrics._resample_daily(pos.abs(), None, pos_change.abs())
        return 100 * pos_change.mean() / pos_abs.mean() 
    
    @staticmethod
    def compute_pnl_per_trade(
        pnl:pd.DataFrame | pd.Series, pos_change:pd.DataFrame | pd.Series
    ) -> pd.Series | float:
        _, pnl, pos_change = Metrics._resample_daily(None, pnl, pos_change.abs())
        return 1e4 * pnl.mean() / pos_change.mean()
    
    @staticmethod
    def compute_mean_returns(
        pos:pd.DataFrame | pd.Series, pnl:pd.DataFrame | pd.Series,
    ) -> pd.Series | float:
        pos_abs, pnl, _ = Metrics._resample_daily(pos.abs(), pnl, None)
        return 252 * pnl.mean() / pos_abs.mean()
    
    @staticmethod
    def compute_maxdrawdown(pnl:pd.DataFrame | pd.Series) -> pd.Series | float:
        _, pnl, _ = Metrics._resample_daily(None, pnl, None)
        return ( pnl.cumsum().cummax() - pnl.cumsum() ).max() / pnl.std()
    
    @staticmethod
    def compute_calamar(pnl:pd.DataFrame | pd.Series) -> pd.Series | float:
        _, pnl, _ = Metrics._resample_daily(None, pnl, None)
        return 16 * Metrics.compute_sharpe(pnl) / Metrics.compute_maxdrawdown(pnl)
    
    @staticmethod
    def compute_sortino(pnl:pd.DataFrame | pd.Series) -> pd.Series | float:
        _, pnl, _ = Metrics._resample_daily(None, pnl, None)
        return 16 * pnl.mean() / pnl[pnl < 0].std()
    
    @staticmethod
    def compute_ftrading(pos:pd.DataFrame | pd.Series) -> pd.Series | float:
        pos_abs, _, _ = Metrics._resample_daily(pos.abs(), None, None)
        return (pos_abs != 0).mean()
    
    @staticmethod
    def compute_win_rate(pnl:pd.DataFrame | pd.Series) -> pd.Series | float:
        _, pnl, _ = Metrics._resample_daily(None, pnl, None)
        return (pnl > 0).sum() / ( (pnl != 0).sum() )
    
    # @staticmethod
    # def compute_long_ratio(pnl:pd.DataFrame | pd.Series, pos:pd.DataFrame | pd.Series) -> pd.Series | float:
    #     return pnl[pos.shift(1) > 0].sum() / pnl.sum()
    
    ### Backtest ###
    
    @staticmethod
    def _compute_metrics_ds(pos:pd.Series, pnl:pd.Series, pos_change:pd.Series) -> pd.Series:
        return pd.Series({
            'eff_sharpe': Metrics.compute_sharpe(pnl[pnl!=0]),
            'raw_sharpe': Metrics.compute_sharpe(pnl),
            'turnover': Metrics.compute_turnover(pos, pos_change),
            'pnl_per_trade': Metrics.compute_pnl_per_trade(pnl, pos_change),
            'mean_returns': Metrics.compute_mean_returns(pnl, pos),
            'maxdrawdown': Metrics.compute_maxdrawdown(pnl),
            'calamar': Metrics.compute_calamar(pnl),
            'sortino': Metrics.compute_sortino(pnl),
            'ftrading': Metrics.compute_ftrading(pos),
            'win_rate': Metrics.compute_win_rate(pnl),
            # 'long_ratio': Metrics.compute_long_ratio(pnl, pos),
            'r_sharpe': Metrics.compute_sharpe(pnl.fillna(0).rolling(252).mean()),
        })
    
    @staticmethod
    def _compute_metrics_df(pos:pd.DataFrame, pnl:pd.DataFrame, pos_change:pd.DataFrame) -> pd.DataFrame:
        tasks = (
            ( pos.loc[:, col], pnl.loc[:, col].fillna(0), pos_change.loc[:, col] ) 
            for col in pos.columns
        )
        with mp.Pool(Metrics.n_jobs) as pool:
            results = pool.starmap(Metrics._compute_metrics_ds, tasks)
        
        return pd.concat({
            col:result_col for col, result_col in zip(pos.columns, results)
        }, axis=1).T.sort_values(by='eff_sharpe', ascending=False)
        
    @staticmethod
    def compute_metrics(
        pos:pd.DataFrame | pd.Series, pnl:pd.DataFrame | pd.Series, pos_change:pd.DataFrame | pd.Series = None
    ) -> pd.DataFrame | pd.Series:
        
        if pos_change is None:
            pos_change = pos.diff().abs() 
        
        pos, pnl, pos_change = Metrics._resample_daily(pos, pnl, pos_change)

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

        pos_abs, pnl, pos_change = Metrics._resample_daily(pos.abs(), pnl, pos_change.abs())

        print( Metrics._compute_metrics_ds(pos_abs.sum(1), pnl.sum(1), pos_change.sum(1)).to_frame('overall').T )

        Utilitaires.plotx( risk * pnl.sum(1).cumsum()                  / pnl.sum(1).std(), title='pnl total' ).show()
        Utilitaires.plotx( risk * Metrics.compute_drawdown(pnl.sum(1)) / pnl.sum(1).std(), title='drawdown' ).show()

        if len(pnl.columns) < 30:
            Utilitaires.plotx( risk * pnl.cumsum() / pnl.std(), title='pnl per asset' ).show()

        return Metrics._compute_metrics_df(pos, pnl, pos_change)
    
    @staticmethod
    def quick_backtest(
        returns:pd.DataFrame | pd.Series, signal:pd.DataFrame, 
        vol:pd.DataFrame = None, bid_ask_spread:pd.DataFrame = None, 
        risk:float = 1, fee_per_transaction:float = 1e-4
    ) -> pd.DataFrame:
        if isinstance(returns, pd.Series):
            returns = returns.to_frame()
        pos = Metrics.compute_position(signal, vol) if vol is not None else Utilitaires.custom_reindex_like(returns)
        pnl = Metrics.compute_pnl(pos, returns)
        if bid_ask_spread is not None:
            pnl -= Metrics.compute_cost(pos.diff().abs(), bid_ask_spread, fee_per_transaction)
        return Metrics.backtest(pos, pnl, pos.diff().abs(), risk)

    @staticmethod
    def monkey_patch_quick_backtest():
        pd.DataFrame.quick_backtest = Metrics.quick_backtest
