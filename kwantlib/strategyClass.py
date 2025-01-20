import pandas as pd 
import numpy  as np 

from typing import Iterable, Callable, Literal, Any

from .utilitaires import Utilitaires
from .operators import Operator
from .tskl_operators import tskl_Operator


############ Strategy Class

class Strategy:

    risk = 1

    def __init__(
            self: 'Strategy', signal:pd.DataFrame, returns:pd.DataFrame, is_vol_target:bool=True
        ) -> None:

        instruments = returns.columns.intersection(
            signal.columns.get_level_values(0).unique()      
        )

        self.signal: pd.DataFrame = signal.loc[:, instruments].replace([np.inf, -np.inf], np.nan).copy()
        self.returns: pd.DataFrame = returns.loc[:, instruments].copy()
        self.is_vol_target = is_vol_target
    
    def _reinit(
            self:'Strategy',signal:pd.DataFrame = None,returns:pd.DataFrame = None, is_vol_target:bool = None
        ) -> 'Strategy':

        return Strategy(
            signal = signal if signal is not None else self.signal.copy(),
            returns= returns if returns is not None else self.returns.copy(),
            is_vol_target = is_vol_target if is_vol_target is not None else self.is_vol_target,
        )
    
    def __getitem__(self:'Strategy', to_keep_list:Iterable[str]) -> 'Strategy':
        return self._reinit(
            returns= self.returns.loc[:, [x for x in self.returns.columns if x in to_keep_list]]
            )
    
    #### Properties
        
    @staticmethod 
    def compute_position(signal:pd.DataFrame, volatility:pd.DataFrame, is_vol_target:bool = True) -> pd.DataFrame:
        signal = signal.reindex(volatility.index, method='ffill').ffill()
        pos = signal.div(volatility, axis = 0, level = 0) if is_vol_target else signal  
        pos = pos.where(Utilitaires.zscore(pos).abs() < 5, np.nan)
        return pd.concat([ 
            pos.loc[:, [col]].reindex(
                volatility.loc[:, col].dropna().index, method='ffill'
            ) # nécessaire pour pas modifier la pos les jours ou l'exchange est close
            for col in volatility.columns
        ], axis = 1).ffill()
    
    @staticmethod
    def compute_pnl(position:pd.DataFrame, returns:pd.DataFrame) -> pd.DataFrame:
        def _pnl(pos:pd.DataFrame, ret:pd.Series) -> pd.DataFrame: 
            # necessaire de faire la multiplication sur les index de ret sinon 
            # introduction de biais style "move the pos at days when you can't"
            pos = pos.reindex(ret.index, method='ffill').ffill()
            return pos.shift(1).multiply(ret, axis=0)
        
        pnl = pd.concat([
            _pnl(position.loc[:, [col]], returns.loc[:, col].dropna()) 
            for col in returns.columns 
        ], axis = 1)

        assert not pnl.apply(np.isinf).any().any(), 'inf in your pnl'

        return pnl
    
    @staticmethod
    def compute_drawdown(pnl:pd.DataFrame) -> pd.DataFrame:
        return - ( pnl.cumsum().cummax() - pnl.cumsum() ) / pnl.std()
    
    @staticmethod
    def compute_cost(pos_change:pd.DataFrame, bid_ask_spread:pd.DataFrame, fee_per_transaction:float = 1e-2) -> pd.DataFrame:
        def _cost(pos_change:pd.Series, bid_ask_spread:pd.Series) -> pd.Series:
            bid_ask_spread = bid_ask_spread.reindex(pos_change.index).ffill().fillna(0)
            return fee_per_transaction * bid_ask_spread * pos_change
        return pd.concat([
            _cost(pos_change.loc[:, col], bid_ask_spread.loc[:, col]) for col in pos_change.columns
        ], axis = 1)
    
    @property
    def volatility(self: 'Strategy') -> pd.DataFrame:
        return self.returns.apply( 
            lambda x: x.dropna().rolling(15).std() 
            )
        # assert all(
        #     vol.loc[:, col].dropna().index.equals(self.returns.loc[:, col].dropna().index[15:]) for col in vol.columns
        # ), 'volatility and returns must have the same index'
    
    @property
    def position(self: 'Strategy') -> pd.DataFrame:
        return Strategy.compute_position(
            self.signal, self.volatility, self.is_vol_target
        )
    
    @property
    def pnl(self: 'Strategy') -> pd.DataFrame:
        return Strategy.compute_pnl(self.position, self.returns)

    @property
    def drawdown(self:'Strategy') -> pd.DataFrame:
        return Strategy.compute_drawdown(self.pnl)
    
    @property
    def return_pnl(self:'Strategy') -> pd.Series:
        pnl = self.pnl.fillna(0).sum(1)
        pos_abs = self.position.abs().sum(1)
        return pnl.div( pos_abs.shift(1) ).fillna(0)
    
    @property
    def compounded_value(self:'Strategy') -> pd.Series:
        return self.return_pnl.apply(lambda x: (1 + x).cumprod())
        
    ### Metrics

    @staticmethod
    def compute_ftrading(pos:pd.DataFrame) -> pd.Series:
        if hasattr(pos.index, 'date'):
            pos = pos.abs().groupby(pos.index.date).mean()
        return (pos > 0).mean()
    
    @staticmethod
    def compute_turnover(pos:pd.DataFrame, pos_change:pd.DataFrame = None) -> pd.Series:
        pos_abs = pos.abs()
        if pos_change is None:
            pos_change = pos.diff().abs()
        if hasattr(pos.index, 'date'):
            pos_abs = pos_abs.groupby(pos_abs.index.date).mean()
            pos_change = pos_change.groupby(pos_change.index.date).sum()
        return 100 * pos_change.mean() / pos_abs.mean() 
    
    @staticmethod
    def compute_pnl_per_trade(pnl:pd.DataFrame, pos_change:pd.DataFrame) -> pd.Series:
        if hasattr(pos_change.index, 'date'):
            pos_change = pos_change.groupby(pos_change.index.date).sum()
            pnl = pnl.groupby(pnl.index.date).sum()
        return 1e4 * pnl.mean() / pos_change.mean()
    
    @staticmethod
    def compute_sharpe(pnl:pd.DataFrame, is_effective:bool = True) -> pd.Series:
        if hasattr(pnl.index, 'date'):
            pnl = pnl.groupby(pnl.index.date).sum()
        if is_effective:
            pnl = pnl[pnl!=0]
        return 16 * pnl.mean() / pnl.std()
    
    @staticmethod
    def compute_maxdrawdown(pnl:pd.DataFrame) -> pd.Series:
        if hasattr(pnl.index, 'date'):
            pnl = pnl.groupby(pnl.index.date).sum()
        return Strategy.compute_drawdown(pnl).min()
    
    @staticmethod
    def compute_metrics(pos:pd.DataFrame, pnl:pd.DataFrame, pos_change:pd.DataFrame = None) -> pd.DataFrame:
        
        if pos_change is None:
            pos_change = pos.diff().abs() 

        metric_dict = {
            'ftrading': Strategy.compute_ftrading(pos),
            'turnover': Strategy.compute_turnover(pos, pos_change),
            'pnl_per_trade': Strategy.compute_pnl_per_trade(pnl, pos_change),
            'eff_sharpe': Strategy.compute_sharpe(pnl),
            'raw_sharpe': Strategy.compute_sharpe(pnl, is_effective=False),
            'r_sharpe': Strategy.compute_sharpe(pnl.fillna(0).rolling(252).mean()),
            'maxdrawdown': Strategy.compute_maxdrawdown(pnl)
        }
        metric_pd = (
            pd.concat(metric_dict, axis=1).sort_values(by='eff_sharpe', ascending=False, axis=0)
            if isinstance(pos, pd.DataFrame) 
            else pd.Series(metric_dict).to_frame('overall').T
        )
        return metric_pd.dropna(how='all', axis=0)
    
    def ftrading(self:'Strategy', training_date:str = None) -> pd.Series:
        pos = self.position.loc[:, training_date:]
        return Strategy.compute_ftrading(pos)
    
    def turnover(self:'Strategy', training_date:str = None) -> pd.Series:
        pos = self.position.loc[:, training_date:]
        return Strategy.compute_turnover(pos)
    
    def pnl_per_trade(self:'Strategy', training_date:str = None) -> pd.Series:
        pos_change = self.position.loc[:, training_date:].diff().abs()
        pnl = self.pnl.loc[:, training_date:].fillna(0)
        return Strategy.compute_pnl_per_trade(pnl, pos_change)
    
    def sharpe(self:'Strategy', training_date:str = None) -> pd.Series:
        pnl = self.pnl.loc[:, training_date:].fillna(0)
        return Strategy.compute_sharpe(pnl)
    
    def raw_sharpe(self:'Strategy', training_date:str = None) -> pd.Series:
        pnl = self.pnl.loc[:, training_date:].fillna(0)
        return Strategy.compute_sharpe(pnl, is_effective=False)
    
    def r_sharpe(self:'Strategy', training_date:str = None) -> pd.Series:
        if hasattr(self.pnl.index, 'date'):
            pnl = self.pnl.groupby(self.pnl.index.date).sum()
        pnl = pnl.loc[:, training_date:].fillna(0).rolling(252).mean()
        return Strategy.compute_sharpe(pnl)
    
    def maxdrawdown(self:'Strategy', training_date:str = None) -> pd.Series:
        pnl = self.pnl.loc[:, training_date:].fillna(0)
        return Strategy.compute_maxdrawdown(pnl)

    def metrics(self:'Strategy', training_date:str = None) -> pd.Series:
        pos = self.position.loc[:, training_date:]
        pnl = self.pnl.loc[:, training_date:].fillna(0)
        pos_change = pos.diff().abs()
        return Strategy.compute_metrics(pos, pnl, pos_change)
    
    ### Backtest

    @staticmethod
    def backtest(pos:pd.DataFrame, pnl:pd.DataFrame, pos_change:pd.DataFrame = None) -> pd.DataFrame:   
        if pos_change is None:
            pos_change = pos.diff().abs()

        if hasattr(pos.index, 'date'):
            pos = pos.groupby(pos.index.date).mean()
            pos_change = pos_change.groupby(pos_change.index.date).sum()
            pnl = pnl.groupby(pnl.index.date).sum()

        pnl_total = pnl.sum(1)
        pos_total = pos.abs().sum(1)
        pos_change_total = pos_change.sum(1)

        print(Strategy.compute_metrics(pos_total, pnl_total, pos_change_total))

        Utilitaires.plotx( Strategy.risk * pnl_total.cumsum() / pnl_total.std(), title='pnl total' ).show()
        Utilitaires.plotx( Strategy.risk * Strategy.compute_drawdown(pnl_total), title='drawdown' ).show()

        if len(pnl.columns.get_level_values(0).unique()) < 30:
            Utilitaires.plotx( 
                Strategy.risk * pnl.cumsum() / pnl.std(), title='pnl per asset' 
            ).show()

        return Strategy.compute_metrics(pos, pnl)

    def show(self:'Strategy', training_date:str=None) -> pd.DataFrame:
        return Strategy.backtest(
            pos = self.position.loc[training_date:, :], 
            pnl = self.pnl.loc[training_date:, :].fillna(0), 
            pos_change = self.position.loc[training_date:, :].diff().abs()
        )

    ### Operators

    def apply(self:'Strategy', func:Callable[[pd.DataFrame, Any], pd.DataFrame], *args, **kwargs) -> 'Strategy':
        return self._reinit(signal = func(self.signal, *args, **kwargs))

    def cross_moving_average(self:'Strategy', *args, **kwargs) -> 'Strategy':
        return self.apply(Operator.cross_moving_average, *args, **kwargs)
    
    def proj(self:'Strategy', *args, **kwargs) -> 'Strategy':
        return self.apply(Operator.proj, *args, **kwargs)
    
    def vote(self:'Strategy', *args, **kwargs) -> 'Strategy':
        return self.apply(Operator.vote, *args, **kwargs)
    
    def ranking(self:'Strategy', *args, **kwargs) -> 'Strategy':
        return self.apply(Operator.ranking, *args, **kwargs)
    
    def markovitz(
        self:'Strategy', method:Literal['minvol', 'maxsharpe'] = 'maxsharpe', level:Literal['cross asset', 'per asset'] = 'cross asset'
    ) -> 'Strategy':
        pnl = self.pnl.shift(1)
        w = Operator.markovitz(pnl, method=method, level=level)
        w = w.reindex(self.signal.index, method='ffill').ffill()
        return self._reinit(signal = self.signal.multiply(w))
    
    def isorisk(self:'Strategy') -> 'Strategy':
        pnl = self.pnl.shift(1)
        w = pnl.apply(lambda x: x.dropna().expanding().std())
        return self._reinit(signal = self.signal.div(w))
    
    def low_sharpe_cutoff(self:'Strategy', threshold:float = 0.3) -> 'Strategy':
        def _sharpe(pnl:pd.Series) -> pd.Series:
            return 16 * pnl.dropna().expanding().mean() / pnl.dropna().expanding().std()
        sharpe = self.pnl.shift(1).apply(_sharpe) 
        return self._reinit(signal = self.signal.where(sharpe < threshold, 0))
    
    ### tskl_operators
    
    def forecast(
        self:'Strategy', lookahead_steps:int = 0, *args, **kwargs
    ) -> 'Strategy':

        target = self.returns.shift(-lookahead_steps)
        features = self.signal
        return self._reinit(
            signal = tskl_Operator.infer(target, features, lookahead_steps = lookahead_steps, *args, **kwargs)
        )

    def residual(
        self:'Strategy', *args, **kwargs
    ) -> 'Strategy':
        target = self.returns
        features = self.signal
        residual_ = target - tskl_Operator.infer(target, features, *args, **kwargs)
        return self._reinit(signal = residual_.apply(lambda x: x.dropna().cumsum()))
    
    def cluster(self:'Strategy', *args, **kwargs) -> 'Strategy':
        raise NotImplementedError('cluster')
