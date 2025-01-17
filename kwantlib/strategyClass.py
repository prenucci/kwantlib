import pandas as pd 
import numpy  as np 

from typing import Iterable, Callable, Literal, Any

from .utilitaires import Utilitaires
from .operator import Operator

from sklearn.linear_model import ElasticNet

############ Strategy Class

class Strategy:

    fee_per_transaction:float = 0

    def __init__(
            self: 'Strategy', signal:pd.DataFrame, returns:pd.DataFrame, spread:pd.DataFrame = None, is_vol_target:bool=True
        ) -> None:

        instruments = returns.columns.intersection(
            signal.columns.get_level_values(0).unique()      
        )

        self.signal: pd.DataFrame = signal.loc[:, instruments].copy()
        self.returns: pd.DataFrame = returns.loc[:, instruments].copy()

        if spread is not None:
            self.spread:pd.DataFrame = spread.loc[:, instruments].copy()
            assert self.spread.index.equals(self.returns.index), 'spread and returns must have the same index'
            assert self.spread.columns.equals(self.returns.columns), 'spread and returns must have the same columns'
        else:  
            self.spread = None

        self.is_vol_target = is_vol_target
    
    def _reinit(
            self:'Strategy',signal:pd.DataFrame = None,returns:pd.DataFrame = None,spread:pd.DataFrame = None,is_vol_target:bool = None
        ) -> 'Strategy':

        return Strategy(
            signal = signal if signal is not None else self.signal.copy(),
            returns= returns if returns is not None else self.returns.copy(),
            spread = spread if spread is not None else self.spread.copy(),
            is_vol_target = is_vol_target if is_vol_target is not None else self.is_vol_target,
        )
    
    def __getitem__(self:'Strategy', to_keep_list:Iterable[str]) -> 'Strategy':
        return self._reinit(
            returns= self.returns.loc[:, [x for x in self.returns.columns if x in to_keep_list]]
            )
        
    @property
    def volatility(self: 'Strategy') -> pd.DataFrame:
        vol = self.returns.apply( 
            lambda x: x.dropna().rolling(15).std() 
            )
        return vol
    
    @staticmethod 
    def compute_position(signal:pd.DataFrame, volatility:pd.DataFrame, is_vol_target:bool = True) -> pd.DataFrame:
        signal = signal.reindex(volatility.index, method='ffill')
        pos = signal.div(volatility, axis = 0, level = 0) if is_vol_target else signal  
        pos = pos.where(Utilitaires.zscore(pos).abs() < 5, np.nan).replace([np.inf, -np.inf], np.nan)
        return pd.concat([ 
            pos.loc[:, [col]].reindex(
                volatility.loc[:, col].dropna().index, method='ffill'
            ) # nécessaire pour pas modifier la pos les jours ou l'exchange est close
            for col in volatility.columns
        ], axis = 1).ffill()
    
    @property
    def position(self: 'Strategy') -> pd.DataFrame:
        return Strategy.compute_position(
            signal = self.signal,
            volatility = self.volatility,
            is_vol_target = self.is_vol_target
        )
    
    @staticmethod
    def compute_pnl(position:pd.DataFrame, returns:pd.DataFrame) -> pd.DataFrame:
        def _pnl(pos:pd.DataFrame, ret:pd.Series) -> pd.DataFrame: 
            # necessaire de faire la multiplication sur les index de ret sinon 
            # introduction de biais style "move the pos at days when you can't"
            pos = pos.reindex(ret.index, method='ffill')
            return pos.shift(1).multiply(ret, axis=0)
        
        pnl = pd.concat([
            _pnl(position.loc[:, [col]], returns.loc[:, col].dropna()) 
            for col in returns.columns 
        ], axis = 1)

        assert not pnl.apply(np.isinf).any().any(), 'inf in your pnl'

        return pnl
    
    @property
    def pnl(self: 'Strategy') -> pd.DataFrame:
        pnl = Strategy.compute_pnl(
                position = self.position,
                returns = self.returns
            )
        return pnl
    
    @staticmethod
    def compute_cost(pos_change:pd.DataFrame, spread:pd.DataFrame, fee_per_transaction:float = 1e-2) -> pd.DataFrame:
        def _cost(pos_change:pd.Series, spread:pd.Series) -> pd.Series:
            spread = spread.reindex(pos_change.index).ffill().fillna(0)
            return fee_per_transaction * spread * pos_change
        return pd.concat([
            _cost(pos_change.loc[:, col], spread.loc[:, col]) for col in pos_change.columns
        ], axis = 1)
    
    @property
    def cost(self:'Strategy') -> pd.DataFrame:
        assert hasattr(self, 'spread'), 'you need to have a spread to compute the cost'
        return Strategy.compute_cost(
            pos_change = self.position.diff().abs(),
            spread = self.spread,
            fee_per_transaction = Strategy.fee_per_transaction
        )
    
    @property
    def net_pnl(self:'Strategy') -> pd.DataFrame:
        return self.pnl.sub(self.cost, axis=0)
    
    @property
    def return_pnl(self:'Strategy') -> pd.Series:
        pnl = self.pnl.fillna(0).sum(1)
        pos_abs = self.position.abs().sum(1)
        return pnl.div(pos_abs.ffill().shift(1)).fillna(0)
    
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
        return ( pnl.cumsum().cummax() - pnl.cumsum() ).max() / pnl.std()
    
    @staticmethod
    def compute_metrics(pos:pd.DataFrame, pnl:pd.DataFrame, pos_change:pd.DataFrame = None) -> pd.DataFrame | pd.Series:
        
        if pos_change is None:
            pos_change = pos.diff().abs() 

        metric = {
            'ftrading': Strategy.compute_ftrading(pos),
            'turnover': Strategy.compute_turnover(pos, pos_change),
            'pnl_per_trade': Strategy.compute_pnl_per_trade(pnl, pos_change),
            'eff_sharpe': Strategy.compute_sharpe(pnl),
            'raw_sharpe': Strategy.compute_sharpe(pnl, is_effective=False),
            'r_sharpe': Strategy.compute_sharpe(pnl.fillna(0).rolling(252).mean()),
            'maxdrawdown': Strategy.compute_maxdrawdown(pnl)
        }
        return (
            pd.concat(metric, axis=1).sort_values(by='eff_sharpe', ascending=False, axis=0) 
            if isinstance(pos, pd.DataFrame) 
            else pd.Series(metric)
            )
    
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
    def backtest(pos:pd.DataFrame, pnl:pd.DataFrame, pos_change:pd.DataFrame = None, cost:pd.DataFrame = None) -> pd.DataFrame:   
        if pos_change is None:
            pos_change = pos.diff().abs()
        if cost is not None:
            pnl = pnl.sub(cost, axis=0)
        pnl_total = pnl.sum(1).to_frame('overall')
        pos_total = pos.abs().sum(1).to_frame('overall')
        pos_change_total = pos_change.sum(1).to_frame('overall')
        print(Strategy.compute_metrics(pos_total, pnl_total, pos_change_total))
        Utilitaires.plotx( pnl_total.cumsum() / pnl_total.std() ).show()
        Utilitaires.plotx( pnl.cumsum() / pnl.std() ).show()
        return Strategy.compute_metrics(pos, pnl)

    def show(self:'Strategy', training_date:str=None) -> pd.DataFrame:
        pnl = self.pnl.fillna(0).loc[training_date:, :]
        pos = self.position.abs()
        pos_change = self.position.diff().abs()
        if hasattr(self, 'spread'):
            cost = self.cost.loc[training_date:, :]
        if hasattr(pos.index, 'date'):
            pos = pos.groupby(pos.index.date).mean()
            pos_change = pos_change.groupby(pos_change.index.date).sum()
            pnl = pnl.groupby(pnl.index.date).sum()
        return Strategy.backtest(pos, pnl, pos_change, cost if hasattr(self, 'spread') else None)

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
    
    ### weighting

    def markovitz(
        self:'Strategy', method:Literal['minvol', 'maxsharpe'] = 'minvol', level:Literal['cross asset', 'per asset'] = 'cross asset'
    ) -> 'Strategy':
        pnl = self.pnl.shift(1)
        w = Operator.markovitz(pnl, method=method, level=level)
        return self._reinit(signal = self.signal.multiply(w))
    
    def isorisk(self:'Strategy') -> 'Strategy':
        pnl = self.pnl.shift(1)
        w = pnl.apply(lambda x: x.dropna().expanding().std())
        return self._reinit(signal = self.signal.div(w))

    ### Cutoffs
    
    def low_sharpe_cutoff(self:'Strategy', threshold:float = 0.3) -> 'Strategy':
        def _sharpe(pnl:pd.Series) -> pd.Series:
            return 16 * pnl.dropna().expanding().mean() / pnl.dropna().expanding().std()
        sharpe = self.pnl.shift(1).apply(_sharpe) < threshold
        return self._reinit(signal = self.signal.where(sharpe, 0))
    
    #### ML

    def forecast(
        self:'Strategy', model:object = ElasticNet(), lookahead_steps:int = 0, *args, **kwargs
    ) -> 'Strategy':

        target = self.returns.shift(-lookahead_steps)
        features = self.signal
        return self._reinit(
            signal = Operator.infer(target, features, model, lookahead_steps = lookahead_steps, *args, **kwargs)
        )

    def residual(
        self:'Strategy', model:object = ElasticNet(), *args, **kwargs
    ) -> 'Strategy':
        target = self.returns
        features = self.signal
        residual_ = target - Operator.infer(target, features, model, *args, **kwargs)
        return self._reinit(signal = residual_.apply(lambda x: x.dropna().cumsum()))
    
    def cluster(self:'Strategy', *args, **kwargs) -> 'Strategy':
        raise NotImplementedError('cluster')
