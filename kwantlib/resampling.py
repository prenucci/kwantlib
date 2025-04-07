import pandas as pd
import numpy as np 

### Resampling ###

def shift_with_sample(
        df:pd.DataFrame | pd.Series, step:int | str = 0
    ) -> pd.DataFrame | pd.Series: 
    """
    Shift the dataframe by a given step after having resampled it daily.
    """
    df_ = df[~df.index.duplicated(keep='first')].sort_index(axis=0)
    return df_.resample('1D').ffill().ffill().shift(step)

def shift_ignoring_nan(
    df:pd.DataFrame | pd.Series, step:int | str = 0
) -> pd.DataFrame | pd.Series:
    """
    Shift the dataframe by a given step ignoring the NaN.
    """
    df_ = df[~df.index.duplicated(keep='first')].sort_index(axis=0)

    match type(df_):
        case pd.Series:
            return df_.dropna().shift(step)
        case pd.DataFrame:
            return pd.concat({
                col: df_[col].dropna().shift(step)
                for col in df_.columns
            }, axis = 1)
        case _:
            raise TypeError(f"df should be a pd.Series or pd.DataFrame not {type(df_)}")
     
### Align Position with Returns ###

def _align_pos_with_returns_ds(
        position:pd.DataFrame | pd.Series, returns:pd.Series, shift:int
    ) -> pd.DataFrame | pd.Series:
    """
    Reindex the position to match the index of not NaN returns for 1 instruments.
    """
    new_index = returns.dropna().index

    # Remove duplicates, shift and fill NaNs
    position = (
        position
        [~position.index.duplicated(keep='first')].sort_index(axis=0)
        .replace([np.inf, -np.inf], np.nan).ffill()
        .shift(shift).fillna(0)
    )

    # Reindex to the cleaned returns index
    return position.reindex(
        new_index, method='ffill'
    ).ffill().fillna(0)

def _align_pos_with_returns_df(
        position:pd.DataFrame, returns:pd.DataFrame, shift:int
    ) -> pd.DataFrame:

    """
    Reindex the position to match the index of not NaN returns for multiples instruments.
    Simply apply the function for each instruments, concatenate the results and ffill.
    """
    return pd.concat([
        _align_pos_with_returns_ds(position.loc[:, [col]], returns.loc[:, col])
        for col in returns.columns 
        if col in position.columns.get_level_values(0)
    ], axis = 1).ffill().fillna(0)

def align_pos_with_returns(
        position:pd.DataFrame | pd.Series, returns:pd.DataFrame | pd.Series, shift:int=0
    ) -> pd.DataFrame | pd.Series:
    """
    This function ensure that: 
    1. There is a position for each index when the exchange is open (when returns is not NaN)
    2. The position can't change when the exchange is closed (when returns is NaN)

    For each instrument:
    1. Get the index when the exchange is open (when returns is not NaN)
    2. Reindex the position on this index and ffill

    Then concatenate the results for the matching instruments.
    """
    match (type(position), type(returns)):
        case (pd.DataFrame | pd.Series, pd.Series):
            return _align_pos_with_returns_ds(position, returns, shift)
        case (pd.DataFrame, pd.DataFrame):
            return _align_pos_with_returns_df(position, returns, shift)
        case _:
            raise ValueError('returns must be a pd.DataFrame or pd.Series')

### Resampling for intraday 

def resample_daily(
    pnl:pd.DataFrame | pd.Series, pos_abs:pd.DataFrame | pd.Series, pos_change:pd.DataFrame | pd.Series
) -> tuple[pd.DataFrame | pd.Series, pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]:
    """
    Resample the pnl, pos_abs and pos_change to daily when intraday. Necessary before computing backtesting.
    """
    
    match (type(pos_abs), type(pnl), type(pos_change)):
        case (pd.Series, pd.Series, pd.Series):
            pass
        case (pd.DataFrame, pd.DataFrame, pd.DataFrame):
            instruments = pnl.columns.intersection(pos_abs.columns).intersection(pos_change.columns)
            pnl, pos_abs, pos_change = pnl.loc[:, instruments], pos_abs.loc[:, instruments], pos_change.loc[:, instruments]
        case _:
            raise ValueError('pos_abs, pnl and pos_change must be of the same type')
    
    match (hasattr(pos_abs.index, 'date'), hasattr(pnl.index, 'date'), hasattr(pos_change.index, 'date')):
        case (True, True, True):
            pos_abs = pos_abs.ffill().groupby(pos_abs.index.date).mean()
            pnl = pnl.fillna(0).groupby(pnl.index.date).sum()
            pos_change = pos_change.fillna(0).groupby(pos_change.index.date).sum()
        case (False, False, False):
            pass
        case _:
            raise ValueError('pos_abs, pnl and pos_change must be either all daily or all intraday')

    return pnl, pos_abs, pos_change
