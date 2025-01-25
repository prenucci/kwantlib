import pandas as pd 
import plotly.express as px 
import multiprocessing as mp 
from typing import Literal

class Utilitaires:

    n_jobs = mp.cpu_count() - 2

    @staticmethod
    def shift_with_sample(
            df:pd.DataFrame, step:int | str = 0
        ) -> pd.DataFrame: 

        df_ = df[~df.index.duplicated(keep='first')]
        return (
            df_
            .dropna(how='all', axis = 1)
            .resample('1D').ffill().ffill()
            .shift(step)
        )
    @staticmethod
    def permute_levels(df:pd.DataFrame, n:int) -> pd.DataFrame: 
        return df.reorder_levels(
            list(range(n, df.columns.nlevels)) + list(range(n)), axis = 1
        ).sort_index(axis=1)
    
    @staticmethod
    def flatten_columns(df:pd.DataFrame) -> pd.DataFrame: 
        df_ = df.copy() 
        if isinstance(df_, pd.DataFrame) and isinstance(df_.columns, pd.MultiIndex):
            df_.columns = df_.columns.map(lambda x: ''.join(str(u) for u in x))
        return df_

    @staticmethod
    def plotx(df:pd.Series | pd.DataFrame, title:str = None):
        match type(df):
            case pd.Series:
                return px.line(df, title=title)
            case pd.DataFrame:
                return px.line(Utilitaires.flatten_columns(df), title=title)
            case _:
                raise TypeError
    
    @staticmethod
    def plotxd(df:pd.Series | pd.DataFrame, title:str = None):
        df_ = df.copy()
        if hasattr(df_.index, 'date'):
            df_ = df_.groupby(df_.index.date).sum()
        return Utilitaires.plotx(df_, title=title)    
    
    @staticmethod
    def _zscore_ds(ds:pd.Series, method:Literal['expanding', 'rolling', 'ewm'] = 'expanding', lookback:int = 252) -> pd.Series:
        match method:
            case 'expanding':
                return (ds - ds.expanding().mean()) / ds.expanding().std()
            case 'rolling':
                return (ds - ds.rolling(lookback).mean()) / ds.rolling(lookback).std()
            case 'ewm':
                return (ds - ds.ewm(lookback).mean()) / ds.ewm(lookback).std()
            case _:
                raise ValueError(f"method should be in ['expanding', 'rolling', 'ewm'] not {method}")

    @staticmethod
    def zscore(
        df: pd.DataFrame | pd.Series, 
        method: Literal['expanding', 'rolling', 'ewm'] = 'expanding', 
        lookback: int = 252, skipna: bool = True
    ) -> pd.DataFrame | pd.Series: 
        
        match type(df):
            case pd.Series:
                zscore = Utilitaires._zscore_ds(df.dropna() if skipna else df, method, lookback)
            case pd.DataFrame:
                zscore = df.apply(lambda x: Utilitaires._zscore_ds(x.dropna() if skipna else x, method, lookback))
            case _:
                raise ValueError(f"df should be a pd.Series or pd.DataFrame not {type(df)}")
        
        return zscore.reindex(df.index).ffill()
    
    @staticmethod
    def _custom_reindex_like_ds(ds:pd.Series | pd.DataFrame, like:pd.Series) -> pd.Series | pd.DataFrame:
        ds_reindexed = ds.reindex(like.dropna().index, method='ffill').ffill().fillna(0)
        assert (
            (ds_reindexed.isna() & like.isna()) | (ds_reindexed.notna() & like.notna())
        ).all(), 'bad reindexing'
        return ds_reindexed
    
    @staticmethod
    def _custom_reindex_like_df(df:pd.DataFrame, like:pd.DataFrame) -> pd.DataFrame:
        assert df.columns.isin(like.columns).all(), 'all columns of df should be in like'

        df_reindexed = pd.concat({
            col: Utilitaires._custom_reindex_like_ds(df.loc[:, col], like.loc[:, col].dropna()) 
            for col in df.columns 
        }, axis=1)

        assert (
            (df_reindexed.isna() & like.isna()) | (df_reindexed.notna() & like.notna())
        ).all(), 'bad reindexing'

        return df_reindexed
    
    @staticmethod
    def custom_reindex_like(df:pd.DataFrame | pd.Series, like:pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        match type(df):
            case pd.Series:
                return Utilitaires._custom_reindex_like_ds(df, like)
            case pd.DataFrame:
                return Utilitaires._custom_reindex_like_df(df, like)
            case _:
                raise ValueError(f"df should be a pd.Series or pd.DataFrame not {type(df)}")

    @staticmethod
    def monkey_patch(): 
        pd.Series.shift_with_sample = Utilitaires.shift_with_sample
        pd.DataFrame.shift_with_sample = Utilitaires.shift_with_sample

        pd.DataFrame.permute_levels = Utilitaires.permute_levels
        pd.DataFrame.flatten_columns = Utilitaires.flatten_columns

        pd.DataFrame.plotx = Utilitaires.plotx
        pd.Series.plotx = Utilitaires.plotx

        pd.DataFrame.plotxd = Utilitaires.plotxd
        pd.Series.plotxd = Utilitaires.plotxd

        pd.Series.zscore = Utilitaires.zscore
        pd.DataFrame.zscore = Utilitaires.zscore

        pd.DataFrame.custom_reindex_like = Utilitaires.custom_reindex_like
        pd.Series.custom_reindex_like = Utilitaires.custom_reindex_like