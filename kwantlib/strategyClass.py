import pandas as pd 
import numpy  as np 

from typing import Iterable, Callable, Literal, Any

from .utilitaires import Utilitaires
from .metrics import Metrics
from .operators import Operator


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
    def return_pnl(self:'Strategy') -> pd.Series:
        return Metrics.compute_returns(self.pnl, self.position)
    
    @property
    def compounded_value(self:'Strategy') -> pd.Series:
        return self.return_pnl.apply(lambda x: (1 + x).cumprod())
        
    ### Metrics
    
    def sharpe(self:'Strategy', training_date:str = None, is_effective:bool = True) -> pd.Series:
        pnl = self.pnl.loc[:, training_date:].fillna(0)
        return Metrics.compute_sharpe(pnl[pnl!=0] if is_effective else pnl)
    
    def turnover(self:'Strategy', training_date:str = None) -> pd.Series:
        pos = self.position.loc[:, training_date:]
        return Metrics.compute_turnover(pos)
    
    def pnl_per_trade(self:'Strategy', training_date:str = None) -> pd.Series:
        pos_change = self.position.loc[:, training_date:].diff().abs()
        pnl = self.pnl.loc[:, training_date:].fillna(0)
        return Metrics.compute_pnl_per_trade(pnl, pos_change)
        
    def mean_returns(self:'Strategy', training_date:str = None) -> pd.Series:
        pnl = self.pnl.loc[:, training_date:].fillna(0)
        pos = self.position.loc[:, training_date:].fillna(0)
        return Metrics.compute_mean_returns(pnl, pos)
    
    def maxdrawdown(self:'Strategy', training_date:str = None) -> pd.Series:
        pnl = self.pnl.loc[:, training_date:].fillna(0)
        return Metrics.compute_maxdrawdown(pnl)
    
    def calamar(self:'Strategy', training_date:str = None) -> pd.Series:
        pnl = self.pnl.loc[:, training_date:].fillna(0)
        return Metrics.compute_calamar(pnl)
    
    def sortino(self:'Strategy', training_date:str = None) -> pd.Series:
        pnl = self.pnl.loc[:, training_date:].fillna(0)
        return Metrics.compute_sortino(pnl)
    
    def ftrading(self:'Strategy', training_date:str = None) -> pd.Series:
        pos = self.position.loc[:, training_date:]
        return Metrics.compute_ftrading(pos)
    
    def r_sharpe(self:'Strategy', training_date:str = None) -> pd.Series:
        pnl = self.pnl.fillna(0).rolling('252D').mean().loc[:, training_date:]
        return Metrics.compute_sharpe(pnl)
    
    def metrics(self:'Strategy', training_date:str = None) -> pd.Series:
        pos = self.position.loc[:, training_date:]
        pnl = self.pnl.loc[:, training_date:].fillna(0)
        pos_change = pos.diff().abs()
        return Metrics.compute_metrics(pos, pnl, pos_change)
    

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
    
    ### Backtest

    def backtest(self:'Strategy', training_date:str=None) -> pd.DataFrame:
        if self.returns.columns.nlevels > 1:
            self = self.proj()

        return Metrics.backtest(
            pos = self.position.loc[training_date:, :], 
            pnl = self.pnl.loc[training_date:, :].fillna(0), 
            pos_change = self.position.loc[training_date:, :].diff().abs(),
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
            vol_target_window = vol_target_window if vol_target_window is not None else self.vol_target_window,
            vol = vol if vol is not None else self.volatility.copy(),
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