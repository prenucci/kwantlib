import pandas as pd 
import numpy  as np 

from typing import Iterable, Callable, Literal, Any

from .utilitaires import Utilitaires
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

        self.volatility = Utilitaires.custom_reindex_like(vol, self.returns).ffill()

    def _reinit(
            self:'Strategy', signal:pd.DataFrame = None, returns:pd.DataFrame = None, 
            vol_target_window:int|str = None, vol:pd.DataFrame = None
        ) -> 'Strategy':

        return Strategy(
            signal = signal if signal is not None else self.signal.copy(),
            returns= returns if returns is not None else self.returns.copy(),
            vol_target_window = vol_target_window if vol_target_window is not None else self.vol_target_window,
            vol = vol if vol is not None else self.volatility.copy()
        )
    
    def __getitem__(self:'Strategy', to_keep_list:Iterable[str]) -> 'Strategy':
        return self._reinit(
            returns= self.returns.loc[:, [x for x in self.returns.columns if x in to_keep_list]]
            )
    
    #### Properties
        
    @staticmethod 
    def compute_position(signal:pd.DataFrame, volatility:pd.DataFrame) -> pd.DataFrame:
        signal = Utilitaires.custom_reindex_like(signal, volatility).ffill()
        pos = signal.div(volatility, axis = 0, level = 0) 
        pos = pos.where(Utilitaires.zscore(pos).abs() < 5, np.nan)
        return Utilitaires.custom_reindex_like(pos, volatility).ffill()
    
    @staticmethod
    def compute_pnl(position:pd.DataFrame, returns:pd.DataFrame) -> pd.DataFrame:
        def _pnl(pos:pd.DataFrame, ret:pd.Series) -> pd.DataFrame: 
            pos = Utilitaires.custom_reindex_like(pos, ret)
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
        
    @property
    def position(self: 'Strategy') -> pd.DataFrame:
        return Strategy.compute_position(self.signal, self.volatility)
    
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
    def compute_sharpe(pnl:pd.DataFrame) -> pd.Series:
        if hasattr(pnl.index, 'date'):
            pnl = pnl.groupby(pnl.index.date).sum()
        return 16 * pnl.mean() / pnl.std()
    
    @staticmethod
    def compute_maxdrawdown(pnl:pd.DataFrame) -> pd.Series:
        if hasattr(pnl.index, 'date'):
            pnl = pnl.groupby(pnl.index.date).sum()
        return - Strategy.compute_drawdown(pnl).min()
    
    @staticmethod
    def compute_calamar(pnl:pd.DataFrame) -> pd.Series:
        if hasattr(pnl.index, 'date'):
            pnl = pnl.groupby(pnl.index.date).sum()
        return 252 * pnl.mean() / Strategy.compute_maxdrawdown(pnl)
        return 16 * Strategy.compute_sharpe(pnl) / Strategy.compute_maxdrawdown(pnl)
    
    @staticmethod
    def compute_sortino(pnl:pd.DataFrame) -> pd.Series:
        if hasattr(pnl.index, 'date'):
            pnl = pnl.groupby(pnl.index.date).sum()
        return 16 * pnl.mean() / pnl[pnl < 0].std()

    @staticmethod
    def compute_metrics(pos:pd.DataFrame, pnl:pd.DataFrame, pos_change:pd.DataFrame = None) -> pd.DataFrame:
        
        if pos_change is None:
            pos_change = pos.diff().abs() 
        
        if isinstance(pos, pd.Series) and isinstance(pnl, pd.Series) and isinstance(pos_change, pd.Series):
            pos, pnl, pos_change = pos.to_frame('overall'), pnl.to_frame('overall'), pos_change.to_frame('overall')

        return (
            pd.concat({
                'ftrading': Strategy.compute_ftrading(pos),
                'turnover': Strategy.compute_turnover(pos, pos_change),
                'pnl_per_trade': Strategy.compute_pnl_per_trade(pnl, pos_change),
                'eff_sharpe': Strategy.compute_sharpe(pnl[pnl!=0]),
                'raw_sharpe': Strategy.compute_sharpe(pnl),
                'r_sharpe': Strategy.compute_sharpe(pnl.fillna(0).rolling(252).mean()),
                'maxdrawdown': Strategy.compute_maxdrawdown(pnl),
                'calamar': Strategy.compute_calamar(pnl),
                'sortino': Strategy.comput_sortino(pnl),
            }, axis=1)
            .sort_values(by='eff_sharpe', ascending=False, axis=0)
            .dropna(how='all', axis=0)
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
    
    def calamar(self:'Strategy', training_date:str = None) -> pd.Series:
        pnl = self.pnl.loc[:, training_date:].fillna(0)
        return Strategy.compute_calamar(pnl)
    
    def sortino(self:'Strategy', training_date:str = None) -> pd.Series:
        pnl = self.pnl.loc[:, training_date:].fillna(0)
        return Strategy.compute_sortino(pnl)
    
    def metrics(self:'Strategy', training_date:str = None) -> pd.Series:
        pos = self.position.loc[:, training_date:]
        pnl = self.pnl.loc[:, training_date:].fillna(0)
        pos_change = pos.diff().abs()
        return Strategy.compute_metrics(pos, pnl, pos_change)

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

    @staticmethod
    def backtest(pos:pd.DataFrame, pnl:pd.DataFrame, pos_change:pd.DataFrame = None) -> pd.DataFrame:   
        if pos_change is None:
            pos_change = pos.diff().abs()

        if hasattr(pos.index, 'date'):
            pos = pos.groupby(pos.index.date).mean()
            pos_change = pos_change.groupby(pos_change.index.date).sum()
            pnl = pnl.groupby(pnl.index.date).sum()

        print(Strategy.compute_metrics(
            pos.abs().sum(1), pnl.sum(1), pos_change.sum(1)
        ))

        Utilitaires.plotx( Strategy.risk * pnl.sum(1).cumsum() / pnl.sum(1).std(), title='pnl total' ).show()
        Utilitaires.plotx( Strategy.risk * Strategy.compute_drawdown(pnl.sum(1)), title='drawdown' ).show()

        if len(pnl.columns) < 30:
            Utilitaires.plotx( Strategy.risk * pnl.cumsum() / pnl.std(), title='pnl per asset' ).show()

        return Strategy.compute_metrics(pos, pnl)

    def show(self:'Strategy', training_date:str=None) -> pd.DataFrame:
        if self.returns.columns.nlevels > 1:
            self = self.proj()

        return Strategy.backtest(
            pos = self.position.loc[training_date:, :], 
            pnl = self.pnl.loc[training_date:, :].fillna(0), 
            pos_change = self.position.loc[training_date:, :].diff().abs()
        )

class StrategyCost(Strategy):

    fee_per_transaction = 1e-4
    
    def __init__(
            self:'StrategyCost', signal:pd.DataFrame, returns:pd.DataFrame, 
            bid_ask_spread:pd.DataFrame, is_vol_target:bool = True,
        ):
        assert bid_ask_spread.columns.equals(returns.columns), 'bid_ask_spread and returns must have the same columns'
        super().__init__(signal, returns, is_vol_target)
        self.bid_ask_spread = bid_ask_spread

    def _reinit(self:'StrategyCost', signal:pd.DataFrame = None, returns:pd.DataFrame = None, bid_ask_spread:pd.DataFrame = None, is_vol_target:bool = None) -> 'StrategyCost':
        return StrategyCost(
            signal = signal if signal is not None else self.signal.copy(),
            returns= returns if returns is not None else self.returns.copy(),
            bid_ask_spread = bid_ask_spread if bid_ask_spread is not None else self.bid_ask_spread.copy(),
            is_vol_target = is_vol_target if is_vol_target is not None else self.is_vol_target,
        )

    @staticmethod
    def compute_cost(
        pos_change:pd.DataFrame, bid_ask_spread:pd.DataFrame, fee_per_transaction:float
    ) -> pd.DataFrame:
        
        def _cost(pos_change_:pd.Series, bid_ask_spread_:pd.Series) -> pd.Series:
            pos_change_ = pos_change_.reindex(bid_ask_spread_.index, method='ffill').ffill().fillna(0)
            return (bid_ask_spread_ / 2 + fee_per_transaction) * pos_change_ 
        
        return pd.concat([
            _cost(pos_change.loc[:, col], bid_ask_spread.loc[:, col].dropna()) 
            for col in bid_ask_spread.columns
        ], axis = 1)

    @property
    def cost(self:'StrategyCost') -> pd.DataFrame:
        return StrategyCost.compute_cost(
            self.position.diff().abs(), self.bid_ask_spread, StrategyCost.fee_per_transaction
        )
    
    @property
    def pnl(self:'StrategyCost') -> pd.DataFrame:
        return super().pnl - self.cost