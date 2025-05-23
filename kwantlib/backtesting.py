import pandas as pd 
import plotly.express as px
from .core import compute_position, compute_pnl
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
    x = y1.index.union(y2.index)
    y1 = y1.reindex(x, method='ffill')
    y2 = y2.reindex(x, method='ffill')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y1, name=title1, line_color='blue'))
    fig.add_trace(go.Scatter(x=x, y=y2, name=title2, line_color='red', yaxis='y2'))
    fig.update_layout(yaxis2={'overlaying': 'y', 'side': 'right'})
    return fig

### Specific metrics 

def _rolling_sharpe(pnl:pd.Series, periods:list[int] = [1/2, 1, 2, 4, 8, 10, 15]) -> pd.DataFrame:
    return pd.concat({
        f'{n}Y': 16 * pnl.rolling(n * 252).mean() / pnl.rolling(n * 252).std()
        for n in periods
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
    flow:pd.DataFrame = None, 
    risk:float = 1, 
    is_aum_cum:bool = False,
    start_date:str = None,
    end_date:str = None,
) -> pd.DataFrame | None:   
        
    """
    Backtest the strategy using the pnl, position and position change.
    Gives a global view (aggregated pnl for all instruments) of the strategy and a per-instrument view.
    """

    if pos is None:
        # Case when no position is provided, just compute pnl related metrics.
        pos = pd.DataFrame(1, index=pnl.index, columns=pnl.columns)
    
    if flow is None:
        # Case when no flow is provided, infer it from the position.
        flow = pos.ffill().fillna(0).diff().abs()

    pnl = pnl.loc[start_date:end_date].fillna(0)
    pos = pos.loc[start_date:end_date].ffill().fillna(0)
    flow = flow.loc[start_date:end_date].fillna(0)

    exposure_total = pos.groupby(level=0, axis=1).sum().abs().sum(1)
    pnl_total = pnl.groupby(level=0, axis=1).sum().sum(1)
    flow_total = flow.groupby(level=0, axis=1).sum().sum(1)

    print(
        compute_metrics(pnl=pnl_total, pos=exposure_total, flow=flow_total).to_frame('overall').T
    )

    px.line(_pnl_cum(pnl_total, risk, is_aum_cum), title='Pnl cum', log_y= is_aum_cum).show()
    px.line(_drawdown(pnl_total, risk, is_aum_cum), title='drawdown').show()
    px.line(_rolling_sharpe(pnl_total), title='rolling sharpe').show()

    gross_exposure = exposure_total / (16 * pnl_total.std())
    rolling_25_days_risk = pnl_total.rolling(25).std() / pnl_total.std()
    
    two_plotx_same_scale(
        gross_exposure, rolling_25_days_risk, title1='gross exposure (std)', title2='rolling 25 days risk (std)'
    ).show()

    if 1 < len(pnl.columns) < 100:
        px.imshow(pnl.corr().fillna(0)).show()

    if len(pnl.columns) < 100:
        scaled_pnl = risk * pnl.cumsum() / pnl_total.std()
        plotx( scaled_pnl.sort_index(axis=1), title='pnl decomposed' ).show()
    
    return pd.concat([
        compute_metrics(pnl=pnl, pos=pos, flow=flow),
        pnl.corrwith(pnl_total).to_frame('corr_with_book')
    ], axis=1) if len(pnl.columns) < 300 else None

def quick_backtest(
        signal:pd.DataFrame, 
        returns:pd.DataFrame, 
        shift:int = 1,
        risk:float = 1, 
        is_aum_cum:bool = False,
        is_roll_diff:bool = False,
        start_date:str = None,
        end_date:str = None,
    ) -> pd.DataFrame | None:
    """
    Quick backtest the strategy using the signal and the returns.
    Signal is shifted before computing the position.
    """
    pos = compute_position(signal, returns, shift=shift)
    pnl = compute_pnl(pos, returns)

    if is_roll_diff:
        prices = returns.fillna(0).cumsum()
        pos = pos.mul(prices, level=0, axis=0)

    flow = pos.diff().abs()

    return backtest(pnl=pnl, pos=pos, flow=flow, risk=risk, is_aum_cum=is_aum_cum, start_date=start_date, end_date=end_date)

def backtest_level(sig:pd.DataFrame, returns:pd.DataFrame, level:int = 0, *args, **kwargs)->pd.DataFrame | None:

    pos = compute_position(sig, returns).ffill().fillna(0)
    pnl = compute_pnl(pos, returns).fillna(0)

    return backtest(
        pnl.groupby(level=level, axis=1).sum(), 
        pos.groupby(level=level, axis=1).sum(), 
        *args, **kwargs
    )
