import pandas as pd 
import numpy  as np 

from typing import Iterable, Callable, Literal, Any

from .utilitaires import Utilitaires
from .core import Core
from .metrics import Metrics
from .operators import Operator
from .weightings import Weighting


############ Strategy Class

class Strategy:

    risk = 1
    vol_target_window = '15D'

    def __init__(
            self:'Strategy', signal:pd.DataFrame, returns:pd.DataFrame, vol:pd.DataFrame = None
        ) -> None:


        if vol is None:
            vol = self.returns.shift(1).apply(
                lambda x: x.dropna().rolling(Strategy.vol_target_window).std()
            )
        
        instruments = (
            signal.columns.get_level_values(0).unique()
            .intersection(returns.columns)
            .intersection(vol.columns)
        )

        self.signal: pd.DataFrame = signal.replace([np.inf, -np.inf], np.nan).loc[:, instruments]
        self.returns: pd.DataFrame = returns.loc[:, instruments]
        self.volatility: pd.DataFrame = vol.loc[:, instruments]

        self.volatility = Utilitaires.custom_reindex_like(self.volatility, self.returns)

    def reinit(
            self:'Strategy', signal:pd.DataFrame = None, returns:pd.DataFrame = None, vol:pd.DataFrame = None
        ) -> 'Strategy':

        return Strategy(
            signal = signal if signal is not None else self.signal.copy(),
            returns= returns if returns is not None else self.returns.copy(),
            vol = vol if vol is not None else self.volatility.copy(),
            vol_target_window= None 
        )
    
    def __getitem__(self:'Strategy', to_keep_list:Iterable[str]) -> 'Strategy':
        return self.reinit(
            returns= self.returns.loc[:, [x for x in self.returns.columns if x in to_keep_list]]
            )
    
    #### Core properties
        
    @property
    def position(self: 'Strategy') -> pd.DataFrame:
        return Core.compute_position(self.signal, self.volatility)
    
    @property
    def pnl(self: 'Strategy') -> pd.DataFrame:
        return Core.compute_pnl(self.position, self.returns)
    
    @property
    def drawdown(self:'Strategy') -> pd.DataFrame:
        return Core.compute_drawdown(self.pnl)
        
    @property
    def ret(self:'Strategy') -> pd.Series:
        return Core.compute_ret(self.position, self.pnl)    
        
    @property
    def compounded_value(self:'Strategy') -> pd.Series:
        return Core.compute_compounded_value(self.position, self.pnl)
        
    ### Operators

    def apply(self:'Strategy', func:Callable[[pd.DataFrame, Any], pd.DataFrame], *args, **kwargs) -> 'Strategy':
        return self.reinit(signal = func(self.signal, *args, **kwargs))

    def cross_moving_average(self:'Strategy', *args, **kwargs) -> 'Strategy':
        return self.apply(Operator.cross_moving_average, *args, **kwargs)
    
    def proj(self:'Strategy', *args, **kwargs) -> 'Strategy':
        return self.apply(Operator.proj, *args, **kwargs)
    
    def vote(self:'Strategy', *args, **kwargs) -> 'Strategy':
        return self.apply(Operator.vote, *args, **kwargs)
    
    def ranking(self:'Strategy', *args, **kwargs) -> 'Strategy':
        return self.apply(Operator.ranking, *args, **kwargs)
    
    ### Weighting

    def weighting(self:'Strategy', func:Callable[[pd.DataFrame, Any], pd.DataFrame], *args, **kwargs) -> 'Strategy':
        pnl = self.pnl_daily.shift(1)
        w = func(pnl, *args, **kwargs)
        w = w.reindex(self.signal.index, method='ffill').ffill()
        return self.reinit(signal = self.signal.multiply(w))

    def markovitz(
        self:'Strategy', 
        method:Literal['minvol', 'maxsharpe'] = 'maxsharpe', 
        level:Literal['cross asset', 'per asset'] = 'cross asset'
    ) -> 'Strategy':
        return self.weighting(Weighting.markovitz, method=method, level=level)
    
    def isorisk(self:'Strategy') -> 'Strategy':
        def _isorisk(pnl:pd.DataFrame) -> pd.DataFrame:
            return pnl.apply(lambda x: 1 / x.dropna().expanding().std()).fillna(0)
        return self.weighting(_isorisk)
    
    def sign_weighting(self:'Strategy') -> 'Strategy':
        def _sign_weighting(pnl:pd.DataFrame) -> pd.DataFrame:
            return pnl.apply(lambda x: x.dropna().expanding().mean().apply(np.sign))
        return self.weighting(_sign_weighting)
    
    ### Cutoffs

    def cutoff(self:'Strategy', func:Callable[[pd.Series, Any], pd.Series], threshold:float = 0.3, *args, **kwargs) -> 'Strategy':
        pnl = self.pnl_daily.shift(1)
        metric = pnl.apply(func, *args, **kwargs)
        return self.reinit(signal = self.signal.where(metric > threshold, 0))

    def low_sharpe_cutoff(self:'Strategy', threshold:float = 0.3) -> 'Strategy':
        def _sharpe(pnl:pd.Series) -> pd.Series:
            return 16 * pnl.dropna().expanding().mean() / pnl.dropna().expanding().std()
        return self.cutoff(_sharpe, threshold)
    
    ### Backtest

    @property
    def pos_abs(self:'Strategy') -> pd.DataFrame:
        pos_abs = self.position.abs().fillna(0)
        if hasattr(pos_abs.index, 'date'):
            pos_abs = pos_abs.groupby(pos_abs.index.date).mean()
        return pos_abs
    
    @property
    def pnl_daily(self:'Strategy') -> pd.DataFrame:
        pnl = self.pnl.fillna(0)
        if hasattr(pnl.index, 'date'):
            return pnl.groupby(pnl.index.date).sum()
        return pnl
    
    @property
    def pos_change(self:'Strategy') -> pd.DataFrame:
        pos_change = self.position.diff().abs().fillna(0)
        if hasattr(pos_change.index, 'date'):
            return pos_change.groupby(pos_change.index.date).sum()
        return pos_change
    
    def metrics(self:'Strategy') -> pd.DataFrame:
        return Metrics.metrics(self.pos_abs, self.pnl_daily, self.pos_change)

    def backtest(self:'Strategy') -> pd.DataFrame:
        if self.returns.columns.nlevels > 1:
            self = self.proj()

        return Metrics.backtest(
            pos = self.pos_abs,
            pnl = self.pnl_daily, 
            pos_change = self.pos_change,
            risk = Strategy.risk
        )
    
    def sharpe(self:'Strategy') -> pd.Series:
        return Metrics.sharpe(self.pnl_daily)
    
    def eff_sharpe(self:'Strategy') -> pd.Series:
        pnl = self.pnl_daily
        return Metrics.sharpe(pnl[pnl != 0])
    
    def turnover(self:'Strategy') -> pd.Series:
        return Metrics.turnover(self.pos_abs, self.pos_change)
    
    def pnl_per_trade(self:'Strategy') -> pd.Series:
        return Metrics.pnl_per_trade(self.pnl_daily, self.pos_change)
    
    def mean_returns(self:'Strategy') -> pd.Series:
        return Metrics.mean_returns(self.pos_abs, self.pnl_daily)
    
    def maxdrawdown(self:'Strategy') -> pd.Series:
        return Metrics.maxdrawdown(self.pnl_daily) 
    
    def sortino(self:'Strategy') -> pd.Series:
        return Metrics.sortino(self.pnl_daily)
    
    def ftrading(self:'Strategy') -> pd.Series:
        return Metrics.ftrading(self.pos_abs)
    
    def calamar(self:'Strategy') -> pd.Series:
        return Metrics.calamar(self.pnl_daily)
    
    def win_rate(self:'Strategy') -> pd.Series:
        return Metrics.win_rate(self.pnl_daily)
    
    def long_ratio(self:'Strategy') -> pd.Series:
        return Metrics.long_ratio(self.pnl_daily, self.pos_abs)
                
class StrategyCost(Strategy):

    fee_per_transaction = 1e-4
    
    def __init__(
            self:'StrategyCost', 
            signal:pd.DataFrame, 
            returns:pd.DataFrame, 
            bid_ask_spread:pd.DataFrame,
            vol:pd.DataFrame = None
        ):
        assert bid_ask_spread.columns.equals(returns.columns), 'bid_ask_spread and returns must have the same columns'
        super().__init__(signal, returns, vol)
        self.bid_ask_spread = bid_ask_spread

    def reinit(
            self:'StrategyCost', signal:pd.DataFrame = None, returns:pd.DataFrame = None, 
            bid_ask_spread:pd.DataFrame = None, vol:pd.DataFrame = None
        ) -> 'StrategyCost':
        return StrategyCost(
            signal = signal if signal is not None else self.signal.copy(),
            returns= returns if returns is not None else self.returns.copy(),
            bid_ask_spread = bid_ask_spread if bid_ask_spread is not None else self.bid_ask_spread.copy(),
            vol = vol if vol is not None else self.volatility.copy(),
        )

    @property
    def cost(self:'StrategyCost') -> pd.DataFrame:
        return Core.compute_cost(
            self.pos_change, 
            self.bid_ask_spread, 
            StrategyCost.fee_per_transaction
        )
        
    @property
    def pnl(self:'StrategyCost') -> pd.DataFrame:
        return super().pnl - self.cost