import pandas as pd 
import plotly.express as px 
from typing import Literal

############ Monkeypatch pandas 

def shift_with_sample(
        df:pd.DataFrame, 
        step:int | str = 0
    ) -> pd.DataFrame: 

    df_ = df[~df.index.duplicated(keep='first')]
    return (
        df_
        .dropna(how='all', axis = 1)
        .resample('1D').ffill()
        .ffill()
        .shift(step)
    )

def permute_levels(df:pd.DataFrame, n:int) -> pd.DataFrame: 
    return df.reorder_levels(
        list(range(n, df.columns.nlevels)) + list(range(n)), axis = 1
    ).sort_index(axis=1)

def flatten_columns(df:pd.DataFrame) -> pd.DataFrame: 
    df_ = df.copy() 
    if isinstance(df_, pd.DataFrame) and isinstance(df_.columns, pd.MultiIndex):
        df_.columns = df_.columns.map(lambda x: ''.join(x))
    return df_

def plotx(df:pd.Series | pd.DataFrame):
    match type(df):
        case pd.Series:
            return px.line(df)
        case pd.DataFrame:
            return px.line(flatten_columns(df))
        case _:
            raise TypeError

def zscore(
    df: pd.DataFrame | pd.Series, 
    method: Literal['expanding', 'rolling', 'ewm'] = 'expanding', 
    lookback: int = 252
) -> pd.DataFrame: 

    window = {
        'expanding': lambda x: x.expanding(),
        'rolling': lambda x: x.rolling(lookback),
        'ewm': lambda x: x.ewm(lookback)
    }[method]

    match type(df):
        case pd.Series:
            zscore = (df - window(df.dropna()).mean()) / window(df.dropna()).std()
        case pd.DataFrame:
            zscore = df.apply(lambda x: (x - window(x.dropna()).mean()) / window(x.dropna()).std())
        case _:
            raise TypeError
    
    return zscore.reindex(df.index).ffill()

def apply_pd_monkeypatch():
    pd.DataFrame.shift_with_sample = shift_with_sample 
    pd.Series.shift_with_sample = shift_with_sample
    pd.DataFrame.permute_levels = permute_levels 
    pd.DataFrame.flatten_columns = flatten_columns

    pd.DataFrame.plotx = plotx 
    pd.Series.plotx = plotx 

    pd.Series.zscore = zscore
    pd.DataFrame.zscore = zscore