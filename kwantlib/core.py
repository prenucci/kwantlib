import pandas as pd 
import numpy as np
import multiprocessing as mp
from .resampling import shift_with_sample, align_pos_with_returns
from .operators import clip_via_zscore

### Compute Pnl ###

def _compute_pnl_ds(position:pd.DataFrame | pd.Series, returns:pd.Series) -> pd.DataFrame | pd.Series:
    """
    Compute the pnl per instrument. 
    Returns is a pd.Series with the returns of an unique instrument. 
    Position can be either a pd.Series (1 instrument per strategies) or a pd.DataFrame (several strategies for 1 instrument) 
    """
    returns_ = returns.dropna()

    position_ = (
        position
        .reindex(returns_.index, method='ffill')
        .ffill()
        .shift(1)
        .fillna(0)
    )

    pnl = position_.mul(returns_, axis=0)   
    assert not pnl.isna().any().any(), 'NaN in your pnl'
    assert not pnl.apply(np.isinf).any().any(), 'inf in your pnl'
    return pnl

def _compute_pnl_df(position:pd.DataFrame, returns:pd.DataFrame) -> pd.DataFrame:
    """ 
    Compute the pnl for multiples instruments. Use multiprocessing with 1 process per instrument. 
    """
    tasks = (
        ( position.loc[:, [col]], returns.loc[:, col]) 
        for col in returns.columns 
        if col in position.columns.get_level_values(0)
    )
    with mp.Pool(mp.cpu_count() - 2) as pool:
        results = pool.starmap(_compute_pnl_ds, tasks)
    pnl = pd.concat(results, axis = 1)
    return pnl.fillna(0)

def compute_pnl(position:pd.DataFrame | pd.Series, returns:pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Compute the pnl from a position and a returns. 
    Position is re-aligned with the returns before computation.
    """
    
    if missing_position := [col for col in returns.columns if col not in position.columns.get_level_values(0)]:
        print(f'missing position: {missing_position}')
    if missing_returns := [col for col in position.columns.get_level_values(0) if col not in returns.columns]:
        print(f'missing returns: {missing_returns}')

    match (type(position), type(returns)):
        case (pd.Series, pd.Series):
            pnl = _compute_pnl_ds(position, returns)
        case (pd.DataFrame, pd.DataFrame | pd.Series):
            pnl = _compute_pnl_df(position, returns)
        case _:
            raise ValueError('returns must be a pd.DataFrame or pd.Series')
    return pnl.fillna(0)


### Compute position 

def compute_position(
        signal:pd.DataFrame, returns:pd.DataFrame, shift:int = 1,
    ) -> pd.DataFrame:
    """ 
    Compute position with:
    1. Shifting 
    2. vol targetting 
    3. zscore clipping 
    4. re-alignement on returns
    """
    volatility = returns.apply(
        lambda x: x.dropna().rolling(25).std()
    ).reindex(signal.index, method='ffill').ffill()
    pos = signal.div(volatility, axis = 0, level = 0).ffill()
    pos = clip_via_zscore(pos, 5, 1008, is_ewm=True)
    pos = shift_with_sample(pos, shift)
    return align_pos_with_returns(pos, returns)

