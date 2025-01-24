import pandas as pd 
import numpy  as np 

from typing import Iterable, Callable, Literal, Any

from .utilitaires import Utilitaires
from .metrics import Metrics
from .operators import Operator, Weighting


############ Strategy Class

class Strategy:

    risk = 1

    def __init__(
            self:'Strategy', 
            signal:pd.DataFrame, 
            returns:pd.DataFrame, 
            vol_target_window:int|str=15, 
            vol:pd.DataFrame = None
        ) -> None:

        instruments = returns.columns.intersection(
            signal.columns.get_level_values(0).unique()      
        )

        self.signal: pd.DataFrame = signal.loc[:, instruments].replace([np.inf, -np.inf], np.nan).copy()
        self.returns: pd.DataFrame = returns.loc[:, instruments].copy()

        if vol is None:
            vol = (
                self.returns.apply(lambda x: x.dropna().rolling(vol_target_window).std())
                if vol_target_window else
                pd.DataFrame(data=1, index=self.returns.index, columns=self.returns.columns)
            )

        self.volatility = Utilitaires.custom_reindex_like(vol, self.returns)

    def _reinit(
            self:'Strategy', signal:pd.DataFrame = None, returns:pd.DataFrame = None, 
            vol_target_window:int|str = None, vol:pd.DataFrame = None
        ) -> 'Strategy':

        return Strategy(
            signal = signal if signal is not None else self.signal.copy(),
            returns= returns if returns is not None else self.returns.copy(),
            vol = vol if vol is not None else self.volatility.copy(),
            vol_target_window= None 
        )
    
    def __getitem__(self:'Strategy', to_keep_list:Iterable[str]) -> 'Strategy':
        return self._reinit(
            returns= self.returns.loc[:, [x for x in self.returns.columns if x in to_keep_list]]
            )
    
    #### Properties
        
    @property
    def position(self: 'Strategy') -> pd.DataFrame:
        return Metrics.compute_position(self.signal, self.volatility)
    
    @property
    def pnl(self: 'Strategy') -> pd.DataFrame:
        return Metrics.compute_pnl(self.position, self.returns)

    @property
    def drawdown(self:'Strategy') -> pd.DataFrame:
        return Metrics.compute_drawdown(self.pnl)
    
    @property
    def ret(self:'Strategy') -> pd.Series:
        return Metrics.compute_ret(self.pnl, self.position)
    
    @property
    def compounded_value(self:'Strategy') -> pd.Series:
        return Metrics.compute_compounded_value(self.position, self.pnl)
        
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
    
    ### Weighting

    def weight(self:'Strategy', func:Callable[[pd.DataFrame, Any], pd.DataFrame], *args, **kwargs) -> 'Strategy':
        pnl = self.pnl.shift(1)
        w = func(pnl, *args, **kwargs)
        w = w.reindex(self.signal.index, method='ffill').ffill()
        return self._reinit(signal = self.signal.multiply(w))

    def markovitz(
        self:'Strategy', method:Literal['minvol', 'maxsharpe'] = 'maxsharpe', level:Literal['cross asset', 'per asset'] = 'cross asset'
    ) -> 'Strategy':
        return self.weight(Weighting.markovitz, method=method, level=level)
    
    def isorisk(self:'Strategy') -> 'Strategy':
        def _isorisk(pnl:pd.DataFrame) -> pd.DataFrame:
            return pnl.apply(lambda x: 1 / x.dropna().expanding().std()).fillna(0)
        return self.weight(_isorisk)
    
    def sign_weighting(self:'Strategy') -> 'Strategy':
        def _sign_weighting(pnl:pd.DataFrame) -> pd.DataFrame:
            return pnl.apply(lambda x: x.dropna().expanding().mean().apply(np.sign))
        return self.weight(_sign_weighting)
    
    ### Cutoffs

    def cutoff(self:'Strategy', func:Callable[[pd.Series, Any], pd.Series], threshold:float = 0.3, *args, **kwargs) -> 'Strategy':
        pnl = self.pnl.shift(1)
        metric = pnl.apply(func, *args, **kwargs)
        return self._reinit(signal = self.signal.where(metric > threshold, 0))

    def low_sharpe_cutoff(self:'Strategy', threshold:float = 0.3) -> 'Strategy':
        def _sharpe(pnl:pd.Series) -> pd.Series:
            return 16 * pnl.dropna().expanding().mean() / pnl.dropna().expanding().std()
        return self.cutoff(_sharpe, threshold)
    
    ### Backtest
            
    def metrics(self:'Strategy') -> pd.DataFrame:
        pos = self.position
        pnl = self.pnl.fillna(0)
        pos_change = pos.diff().abs()
        return Metrics.compute_metrics(pos, pnl, pos_change)

    def backtest(self:'Strategy') -> pd.DataFrame:
        if self.returns.columns.nlevels > 1:
            self = self.proj()

        return Metrics.backtest(
            pos = self.position,
            pnl = self.pnl.fillna(0), 
            pos_change = self.position.diff().abs(),
            risk = Strategy.risk
        )

class StrategyCost(Strategy):

    fee_per_transaction = 1e-4
    
    def __init__(
            self:'StrategyCost', signal:pd.DataFrame, returns:pd.DataFrame, 
            bid_ask_spread:pd.DataFrame, vol_target_window:int|str=15, 
            vol:pd.DataFrame = None
        ):
        assert bid_ask_spread.columns.equals(returns.columns), 'bid_ask_spread and returns must have the same columns'
        super().__init__(signal, returns, vol_target_window, vol)
        self.bid_ask_spread = bid_ask_spread

    def _reinit(
            self:'StrategyCost', signal:pd.DataFrame = None, returns:pd.DataFrame = None, 
            bid_ask_spread:pd.DataFrame = None, vol_target_window:int|str = None, 
            vol:pd.DataFrame = None
        ) -> 'StrategyCost':
        return StrategyCost(
            signal = signal if signal is not None else self.signal.copy(),
            returns= returns if returns is not None else self.returns.copy(),
            bid_ask_spread = bid_ask_spread if bid_ask_spread is not None else self.bid_ask_spread.copy(),
            vol = vol if vol is not None else self.volatility.copy(),
            vol_target_window = None,
        )

    @property
    def cost(self:'StrategyCost') -> pd.DataFrame:
        return Metrics.compute_cost(
            self.position.diff().abs(), 
            self.bid_ask_spread, 
            StrategyCost.fee_per_transaction
        )
    
    @property
    def pnl(self:'StrategyCost') -> pd.DataFrame:
        return super().pnl - self.cost