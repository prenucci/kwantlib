import pandas as pd 
import multiprocessing as mp 
import plotly.express as px
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
    def maxdrawdown(pnl:pd.DataFrame | pd.Series) -> pd.Series | float:
        return (
            pnl.cumsum().cummax() - pnl.cumsum()
        ).max() / pnl.std()
    
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
    
    @staticmethod
    def unlevered_mean_return(pnl:pd.DataFrame | pd.Series, pos:pd.DataFrame | pd.Series) -> pd.Series | float:
        return 100 * 252 * pnl.mean() / pos.abs().mean()
    
    @staticmethod
    def unlevered_std(pnl:pd.DataFrame | pd.Series, pos:pd.DataFrame | pd.Series) -> pd.Series | float:
        return 100 * 16 * pnl.std() / pos.abs().mean()
    
    ### Backtest ###
    
    @staticmethod
    def _metrics_ds(pos:pd.Series, pnl:pd.Series, pos_change:pd.Series) -> pd.Series:
        return pd.Series({
            'eff_sharpe': Metrics.sharpe(pnl[pnl!=0]),
            'raw_sharpe': Metrics.sharpe(pnl),
            'turnover': Metrics.turnover(pos, pos_change),
            'pnl_per_trade': Metrics.pnl_per_trade(pnl, pos_change),
            'maxdrawdown': Metrics.maxdrawdown(pnl),
            'calamar': Metrics.calamar(pnl),
            'sortino': Metrics.sortino(pnl),
            'ftrading': Metrics.ftrading(pos),
            'win_rate': Metrics.win_rate(pnl),
            'r_sharpe': Metrics.sharpe(pnl.fillna(0).rolling(252).mean()) / 16,
            'unlevered_mean_return': Metrics.unlevered_mean_return(pnl, pos),
            'unlevered_std': Metrics.unlevered_std(pnl, pos)
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
    def _resample_daily(
        pos_abs:pd.DataFrame | pd.Series, pnl:pd.DataFrame | pd.Series, pos_change:pd.DataFrame | pd.Series
    ) -> tuple[pd.DataFrame | pd.Series, pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]:
        
        match (type(pos_abs), type(pnl), type(pos_change)):
            case (pd.Series, pd.Series, pd.Series):
                pass
            case (pd.DataFrame, pd.DataFrame, pd.DataFrame):
                instruments = pnl.columns.intersection(pos_abs.columns).intersection(pos_change.columns)
                pos_abs, pnl, pos_change = pos_abs.loc[:, instruments], pnl.loc[:, instruments], pos_change.loc[:, instruments]
            case _:
                raise ValueError('pos_abs, pnl and pos_change must be of the same type')
        
        match (hasattr(pos_abs.index, 'date'), hasattr(pnl.index, 'date'), hasattr(pos_change.index, 'date')):
            case (True, True, True):
                pos_abs = pos_abs.ffill().groupby(pos_abs.index.date).mean()
                pnl = pnl.fillna(0).groupby(pnl.index.date).sum()
                pos_change = pos_change.fillna(0).groupby(pos_change.index.date).sum()
            case (False, False, False):
                pass
            case _:
                raise ValueError('pos_abs, pnl and pos_change must be either all daily or all intraday')

        return pos_abs, pnl, pos_change
        
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
                return Metrics._metrics_df(pos_abs, pnl, pos_change)
            case _:
                raise ValueError('pos, pnl and pos_change must be of the same type')
        
    @staticmethod
    def rolling_sharpe(pnl:pd.Series, periods:list[int] = [1/2, 1, 2, 4, 8]) -> pd.DataFrame:
        return pd.concat({
            f'{n}D': 16 * pnl.rolling(n).mean() / pnl.rolling(n).std()
            for n in [int(252 * x) for x in periods]
        } | {
            'expanding': 16 * pnl.expanding(min_periods=504).mean() / pnl.expanding(min_periods=504).std()
        }, axis=1).ffill()
    
    @staticmethod
    def pnl_cum(pnl:pd.Series, risk:float = 1, is_aum_cum:bool = False) -> pd.Series:
        pnl_scaled = (risk / 16) * pnl / pnl.std()
        if is_aum_cum:
            return (1 + ( pnl_scaled / 100 )).cumprod()
        else:
            return pnl_scaled.cumsum()
    
    @staticmethod
    def drawdown(pnl:pd.Series, risk:float = 1, is_aum_cum:bool = False) -> pd.Series:
        pnl_cum = Metrics.pnl_cum(pnl, risk, is_aum_cum)
        if is_aum_cum:
            return ( pnl_cum - pnl_cum.cummax() ) / pnl_cum.cummax()
        else:
            return ( pnl_cum - pnl_cum.cummax() )
        
    @staticmethod
    def backtest(
        pnl:pd.DataFrame, pos:pd.DataFrame = None, 
        pos_change:pd.DataFrame = None, 
        risk:float = 1, is_aum_cum:bool = False
    ) -> pd.DataFrame:   
        
        if pos is None:
            pos = pd.DataFrame(1, index=pnl.index, columns=pnl.columns)
        
        if pos_change is None:
            pos_change = pos.diff().abs()

        pos_abs, pnl, pos_change = Metrics._resample_daily(pos.abs(), pnl, pos_change)
        pos_abs_total, pnl_total, pos_change_total = pos_abs.sum(1), pnl.sum(1), pos_change.sum(1)
        
        print(Metrics.metrics(pos_abs_total, pnl_total, pos_change_total).to_frame('overall').T)

        px.line(Metrics.pnl_cum(pnl_total, risk, is_aum_cum), title='Pnl cum', log_y= is_aum_cum).show()
        px.line(Metrics.drawdown(pnl_total, risk, is_aum_cum), title='drawdown').show()
        px.line(Metrics.rolling_sharpe(pnl_total), title='rolling sharpe').show()
        Utilitaires.clustermap(pnl.corr().fillna(0)).show()

        if len(pnl.columns) < 30:
            Utilitaires.plotx( risk * pnl.cumsum() / pnl.std(), title='pnl decomposed' ).show()

        return pd.concat([
            Metrics.metrics(pos_abs, pnl, pos_change),
            pnl.corrwith(pnl_total).to_frame('corr_with_book')
        ], axis=1)
    
    @staticmethod
    def quick_backtest(
        returns:pd.DataFrame, signal:pd.DataFrame, 
        vol:pd.DataFrame = None, 
        risk:float = 1, is_aum_cum:bool = False, 
    ) -> pd.DataFrame:
        pos = Core.compute_position(signal, vol) if vol is not None else signal
        pnl = Core.compute_pnl(pos, returns)
        return Metrics.backtest(pnl=pnl, pos=pos, risk=risk, is_aum_cum=is_aum_cum)








