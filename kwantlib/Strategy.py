import pandas as pd 
import numpy  as np 

from typing import Iterable, Callable

from .pandasMonkeypatch import apply_pd_monkeypatch
from .operators import proj, vote, cross_moving_average, markovitz_maxsharpe, ranking #, infer

apply_pd_monkeypatch()

############ Strategy Class

class Strategy:

    def __init__(
        self: 'Strategy', 
        signal:pd.DataFrame,
        returns:pd.DataFrame,
        spread:pd.DataFrame = None,
        is_vol_target:bool=True,
        fee_per_transaction:float = 5e-1,
    ):

        instruments = returns.columns.intersection(
            signal.columns.get_level_values(0).unique()      
        )

        if len(instruments) == 0: #All the same signal for all the assets
            signal = pd.concat({col:signal for col in self.returns.columns}, axis=1)
            instruments = returns.columns

        self.signal: pd.DataFrame = signal.loc[:, instruments].copy()
        self.returns: pd.DataFrame = returns.loc[:, instruments].copy()
        if spread is not None:
            self.spread:pd.DataFrame = spread.loc[:, instruments].copy()
        self.is_vol_target = is_vol_target
        self.fee_per_transaction = fee_per_transaction
    
    def _reinit(
        self:'Strategy',
        signal:pd.DataFrame = None,
        returns:pd.DataFrame = None,
        spread:pd.DataFrame = None,
        is_vol_target:bool = None,
        fee_per_transaction:float = None,
    ) -> 'Strategy':

        return Strategy(
            signal = signal if signal is not None else self.signal.copy(),
            returns= returns if returns is not None else self.returns.copy(),
            spread = spread if spread is not None else self.spread.copy(),
            is_vol_target = is_vol_target if is_vol_target is not None else self.is_vol_target,
            fee_per_transaction = fee_per_transaction if fee_per_transaction is not None else self.fee_per_transaction,
        )
    
    def __getitem__(self:'Strategy', to_keep_list:Iterable[str]) -> 'Strategy':
        return self._reinit(
            returns= self.returns.loc[:, [x for x in self.returns.columns if x in to_keep_list]]
            )
    
    #### Properties
    
    @property
    def volatility(self: 'Strategy') -> pd.DataFrame:
        return self.returns.apply( 
            lambda x: x.dropna().rolling(15).std() 
            )
    
    @staticmethod 
    def compute_position(signal:pd.DataFrame, volatility:pd.DataFrame, is_vol_target:bool = True) -> pd.DataFrame:
        signal = signal.reindex(volatility.index).ffill().ffill()
        pos = signal.div(volatility, axis = 0, level = 0) if is_vol_target else signal  
        pos = pos.where(pos.zscore().abs() < 5, np.nan).ffill()
        return pd.concat([ 
            pos.loc[:, [col]].reindex(
                volatility.loc[:, col].dropna().index
            ).ffill() # nécessaire pour pas modifier la pos les jours ou l'exchange est close
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
            return pos.shift(1).multiply(ret, axis=0)
        
        pnl = pd.concat([
            _pnl(position.loc[:, [col]], returns.loc[:, col].dropna()) for col in returns.columns 
        ], axis = 1).fillna(0)

        assert not pnl.apply(np.isinf).any().any(), 'inf in your pnl'

        return pnl.fillna(0)    
    
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
            fee_per_transaction = self.fee_per_transaction
        )
    
    @property
    def net_pnl(self:'Strategy') -> pd.DataFrame:
        return self.pnl.sub(self.cost, axis=0)
    
    @property
    def return_pnl(self:'Strategy') -> pd.Series:
        pnl, pos = self.pnl.fillna(0).sum(1), self.position.abs().sum(1)
        return pnl.div(pos.ffill().shift(1)).where(pos != 0, 0)
    
    @property
    def compounded_value(self:'Strategy') -> pd.Series:
        return self.return_pnl.apply(lambda x: (1 + x).cumprod())
        
    ### Metrics
    @staticmethod
    def compute_ftrading(pos:pd.DataFrame) -> pd.Series:
        if hasattr(pos.index, 'date'):
            pos_abs = pos.abs().groupby(pos.index.date).mean()
        return (pos_abs > 0).mean()
    
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
    def compute_pnl_per_trade(pnl:pd.DataFrame, pos_change:pd.DataFrame = None) -> pd.Series:
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
        ( pnl_total.cumsum() / pnl_total.std() ).plotx().show()
        ( pnl.cumsum() / pnl.std() ).plotx().show()
        return Strategy.compute_metrics(pos, pnl)

    def show(self:'Strategy', training_date:str=None) -> pd.DataFrame:
        pnl = self.pnl.fillna(0).loc[training_date:, :]
        pos = self.position.abs()
        pos_change = pos.diff().abs()
        if hasattr(self, 'spread'):
            cost = self.cost.loc[training_date:, :]
        if hasattr(pos.index, 'date'):
            pos = pos.groupby(pos.index.date).mean()
            pos_change = pos_change.groupby(pos_change.index.date).sum()
            pnl = pnl.groupby(pnl.index.date).sum()
        return Strategy.backtest(pos, pnl, pos_change, cost if hasattr(self, 'spread') else None)

    ### Operators

    def apply(self:'Strategy', func:Callable[[pd.DataFrame], pd.DataFrame], *args, **kwargs) -> 'Strategy':
        return self._reinit(
            signal = func(self.signal, *args, **kwargs)
            )

    def cross_moving_average(self:'Strategy', *args, **kwargs) -> 'Strategy':
        return self.apply(cross_moving_average, *args, **kwargs)
    
    def proj(self:'Strategy', *args, **kwargs) -> 'Strategy':
        return self.apply(proj, *args, **kwargs)
    
    def vote(self:'Strategy', *args, **kwargs) -> 'Strategy':
        return self.apply(vote, *args, **kwargs)
    
    def ranking(self:'Strategy', *args, **kwargs) -> 'Strategy':
        return self.apply(ranking, *args, **kwargs)
    
    ### weighting

    def weighting(self:'Strategy', func:Callable[[pd.DataFrame], pd.DataFrame], *args, **kwargs) -> 'Strategy':
        pnl_train = self.pnl.shift(1)
        weights = func(pnl_train, *args, **kwargs).ffill(0)
        return self._reinit(
            signal = self.signal.multiply(weights, axis = 0)
            )
    
    def low_sharpe_cutoff(self:'Strategy', threshold:float = 0.3) -> 'Strategy':
        def _sharpe(pnl:pd.Series) -> pd.Series:
            return 16 * pnl.expanding().mean() / pnl.expanding().std()
        return self.weighting(lambda pnl: _sharpe(pnl) > threshold)

    def markovitz(self:'Strategy', *args, **kwargs) -> 'Strategy':
        return self.weighting(markovitz_maxsharpe, *args, **kwargs)
    
    def isorisk(self:'Strategy') -> 'Strategy':
        return self.weighting(lambda pnl: 1 / pnl.dropna().expanding().std())

