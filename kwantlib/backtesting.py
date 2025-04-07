import pandas as pd 
import plotly.express as px
from .core import align_pos_with_returns, compute_pnl
from .metrics import compute_metrics 
import plotly.graph_objects as go

### Plot with plotly ###

def flatten_index(df:pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    df_ = df.copy()
    if isinstance(df_.index, pd.MultiIndex):
        df_.index = df_.index.map(lambda x: ''.join(str(u) for u in x))
    if isinstance(df_.columns, pd.MultiIndex):
        df_.columns = df_.columns.map(lambda x: ''.join(str(u) for u in x))
    return df_

def plotx(df:pd.Series | pd.DataFrame, title:str = None):
    """
    Plot the dataframe using plotly.express
    """

    match type(df):
        case pd.Series:
            return px.line(df, title=title)
        case pd.DataFrame:
            return px.line(flatten_index(df), title=title)
        case _:
            raise TypeError
        
def two_plotx_same_scale(y1:pd.Series, y2:pd.Series, title1:str = None, title2:str = None):
    assert y1.index.equals(y2.index)
    x = y1.index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y1, name=title1, line_color='blue'))
    fig.add_trace(go.Scatter(x=x, y=y2, name=title2, line_color='red', yaxis='y2'))
    fig.update_layout(yaxis2={'overlaying': 'y', 'side': 'right'})
    return fig

### Specific metrics 

def _rolling_sharpe(pnl:pd.Series, periods:list[int] = [1/2, 1, 2, 4, 8, 10, 15]) -> pd.DataFrame:
    return pd.concat({
        f'{n}D': 16 * pnl.rolling(n).mean() / pnl.rolling(n).std()
        for n in [int(252 * x) for x in periods]
    } | {
        'expanding': 16 * pnl.expanding(min_periods=1008).mean() / pnl.expanding(min_periods=1008).std(),
        'overall': pd.Series(16 * pnl.mean() / pnl.std(), index=pnl.index), 
    }, axis=1).ffill()

def _pnl_cum(pnl:pd.Series, risk:float = 1, is_aum_cum:bool = False) -> pd.Series:
    pnl_scaled = (risk / 16) * ( pnl / pnl.std() )
    if is_aum_cum:
        return (1 + ( pnl_scaled / 100 )).cumprod()
    else:
        return pnl_scaled.cumsum()

def _drawdown(pnl:pd.Series, risk:float = 1, is_aum_cum:bool = False) -> pd.Series:
    pnl_cum_ = _pnl_cum(pnl, risk, is_aum_cum)
    if is_aum_cum:
        return ( pnl_cum_ - pnl_cum_.cummax() ) / pnl_cum_.cummax()
    else:
        return ( pnl_cum_ - pnl_cum_.cummax() )

### Backtest ###

def backtest(
    pnl:pd.DataFrame, 
    pos:pd.DataFrame = None, 
    pos_change:pd.DataFrame = None, 
    risk:float = 1, is_aum_cum:bool = False
) -> pd.DataFrame:   
        
    """
    Backtest the strategy using the pnl, position and position change.
    Gives a global view (aggregated pnl for all instruments) of the strategy and a per-instrument view.
    """
    
    if pos is None:
        # Case when no position is provided, just compute pnl related metrics.
        pos = pd.DataFrame(1, index=pnl.index, columns=pnl.columns)
    
    if pos_change is None:
        pos_change = pos.ffill().fillna(0).diff().abs()

    pos_abs_total = pos.ffill().fillna(0).abs().sum(1)
    pnl_total = pnl.fillna(0).sum(1)
    pos_change_total = pos_change.fillna(0).sum(1)

    print(
        compute_metrics(pnl=pnl_total, pos=pos_abs_total, pos_change=pos_change_total).to_frame('overall').T
    )

    px.line(_pnl_cum(pnl_total, risk, is_aum_cum), title='Pnl cum', log_y= is_aum_cum).show()
    px.line(_drawdown(pnl_total, risk, is_aum_cum), title='drawdown').show()
    px.line(_rolling_sharpe(pnl_total), title='rolling sharpe').show()


    gross_exposure = pos_abs_total / (16 * pnl_total.std())
    rolling_25_days_risk = pnl_total.rolling(25).std() / pnl_total.std()

    two_plotx_same_scale(
        gross_exposure, rolling_25_days_risk, title1='gross exposure (std)', title2='rolling 25 days risk (std)'
    ).show()

    if len(pnl.columns) > 1:
        px.imshow(pnl.corr().fillna(0)).show()

    if len(pnl.columns) < 70:
        plotx( risk * pnl.cumsum() / pnl_total.std(), title='pnl decomposed' ).show()

    return pd.concat([
        compute_metrics(pnl=pnl, pos=pos, pos_change=pos_change),
        pnl.corrwith(pnl_total).to_frame('corr_with_book')
    ], axis=1)

def quick_backtest(signal:pd.DataFrame, returns:pd.DataFrame, risk:float = 1, is_aum_cum:bool = False) -> pd.DataFrame:
    """
    Quick backtest the strategy using the signal and the returns.
    Signal is first aligned with the returns before backtesting. 
    """
    pos = align_pos_with_returns(signal, returns)
    pos_change = pos.diff().abs()
    pnl = compute_pnl(pos, returns)
    return backtest(pnl=pnl, pos=pos, pos_change=pos_change, risk=risk, is_aum_cum=is_aum_cum)



