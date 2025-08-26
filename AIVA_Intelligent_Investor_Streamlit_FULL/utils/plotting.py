
import plotly.graph_objects as go
import pandas as pd

def price_chart(df: pd.DataFrame, title: str = "Koersgrafiek"):
    fig = go.Figure()
    if df is None or df.empty:
        fig.update_layout(title="Geen data", template="plotly_white", height=380)
        return fig
    fig.add_trace(go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Prijs"
    ))
    fig.update_layout(title=title, template="plotly_white", height=420, xaxis_rangeslider_visible=False)
    return fig
