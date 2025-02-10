import pandas as pd 
import plotly.express as px 
import seaborn as sns
import multiprocessing as mp 
from typing import Callable, Any

class Utilitaires:

    n_jobs = mp.cpu_count() - 2

    @staticmethod
    def shift_with_sample(
            df:pd.DataFrame, step:int | str = 0
        ) -> pd.DataFrame: 

        df_ = df[~df.index.duplicated(keep='first')]
        return df_.dropna(how='all', axis = 1).resample('1D').ffill().ffill().shift(step)
    
    @staticmethod
    def shift_ignoring_nan(
        df:pd.DataFrame, step:int | str = 0
    ) -> pd.DataFrame:
        match type(df):
            case pd.Series:
                return df.dropna().shift(step)
            case pd.DataFrame:
                return df.dropna(how='all', axis = 1).apply(lambda x: x.dropna().shift(step))
            case _:
                raise ValueError(f"df should be a pd.Series or pd.DataFrame not {type(df)}")
        
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
    def clustermap(corr:pd.DataFrame):
        return sns.clustermap(
            data=corr,metric='euclidean',
            vmin=-1, vmax=1, cmap='coolwarm', 
            figsize=(8, 4), annot=True, fmt='.2f',
        ).figure
    
    @staticmethod
    def _custom_reindex_like_ds(ds:pd.Series | pd.DataFrame, like:pd.Series) -> pd.Series | pd.DataFrame:
        return ds.reindex(like.dropna().index, method='ffill').ffill().fillna(0)
    
    @staticmethod
    def _custom_reindex_like_df(df:pd.DataFrame, like:pd.DataFrame) -> pd.DataFrame:
        assert df.columns.get_level_values(0).isin(like.columns).all(), 'all columns of df should be in like'

        return pd.concat({
            col: Utilitaires._custom_reindex_like_ds(df.loc[:, col], like.loc[:, col].dropna()) 
            for col in df.columns.get_level_values(0).unique() 
        }, axis=1)
    
    @staticmethod
    def custom_reindex_like(
        df:pd.DataFrame | pd.Series, like:pd.DataFrame | pd.Series
    ) -> pd.DataFrame | pd.Series:
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

        pd.Series.shift_ignoring_nan = Utilitaires.shift_ignoring_nan
        pd.DataFrame.shift_ignoring_nan = Utilitaires.shift_ignoring_nan

        pd.DataFrame.permute_levels = Utilitaires.permute_levels
        pd.DataFrame.flatten_columns = Utilitaires.flatten_columns

        pd.DataFrame.plotx = Utilitaires.plotx
        pd.Series.plotx = Utilitaires.plotx

        pd.DataFrame.plotxd = Utilitaires.plotxd
        pd.Series.plotxd = Utilitaires.plotxd

        pd.DataFrame.custom_reindex_like = Utilitaires.custom_reindex_like
        pd.Series.custom_reindex_like = Utilitaires.custom_reindex_like

    @staticmethod
    def from_Series_operator_to_DataFrame_operator(
        func:Callable[[pd.Series, Any], pd.Series], 
    )->Callable[[pd.DataFrame, Any], pd.DataFrame]:
        
        def apply_func(sig:pd.Series, args, kwargs) -> pd.Series:
            return func(sig, *args, **kwargs)
            
        def func_decorated(
                signal:pd.DataFrame, *args, **kwargs
            ) -> pd.DataFrame:

            tasks = ( (signal.loc[:, col].dropna(), args, kwargs) for col in signal.columns)

            with mp.Pool(Utilitaires.n_jobs) as pool:
                results = pool.starmap(apply_func, tasks)
                
            return pd.concat({
                col: res for col, res in zip(signal.columns, results)
            }, axis=1)
        
        return func_decorated