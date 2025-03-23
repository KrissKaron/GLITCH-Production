from __path__ import *
import sys
import asyncio
import threading
import time
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from datetime import datetime, timedelta
from stable_baselines3 import PPO

# Proprietary Imports
from async_methods import *
from _pivot import *
from _deltaT import *
from _impact import *
from _RL import *
from _amplitude import *
from _signal_updater import *
from _plottation_device import *

# PATH Configuration
sys.path.extend([f'{PATH_WAR}', f'{PATH_CLASS}'])

# Async Data Processing
async def main():
    model = await PURPLE_train_or_load_model()
    while True:
        tasks = [
            fetch_general_news(),
            fetch_regulatory_news(),
            fetch_whale_alerts(),
            fetch_economic_news(),
            fetch_on_chain_data(),
            fetch_mining_data(),
            fetch_binance_klines(),
            excavate_score_compact(),
            run_pivot_detection(PATH_KLINES, PATH_PIVOTS),
            PURPLE_run_period_extraction(PATH_PIVOTS, PATH_PIVOT_PERIODS),
            PURPLE_run_impact_analysis(PATH_NEWS, PATH_NEWS_IMPACT),
            PURPLE_adjust_news_impact(PATH_PIVOT_PERIODS, PATH_NEWS_IMPACT, model, PATH_NEWS_PERIODS),
            PURPLE_run_price_estimation(PATH_KLINES, PATH_NEWS_PERIODS, PATH_DAMPED),
            PURPLE_update_signals(PATH_DAMPED, PATH_FUTURE_TRADES),
            TRADE_future(PATH_FUTURE_TRADES, PATH_KLINES),
            TRADE_MikeRoss(PATH_KLINES, PATH_FUTURE_TRADES)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, result in enumerate(results):
            print(f"Task {idx+1} {'failed' if isinstance(result, Exception) else 'completed successfully'}: {result}")
        print("Waiting for next run...")
        await asyncio.sleep(60)

# Data Loading
future_trades_df = pd.read_csv(PATH_FUTURE_TRADES)
klines_df = pd.read_csv(PATH_KLINES)
future_trades_df['timestamp'] = pd.to_datetime(future_trades_df['timestamp'])
klines_df['timestamp'] = pd.to_datetime(klines_df['timestamp'])
klines_df.dropna(subset=['timestamp'], inplace=True)
data = []  # Data storage

# Data Synchronization and Storage
def update_data():
    global data
    merged_df = pd.merge_asof(
        future_trades_df.sort_values('timestamp'),
        klines_df.sort_values('timestamp'),
        on='timestamp'
    )
    for _, row in merged_df.iterrows():
        data.append({
            'timestamp': row['timestamp'],
            'price': row['price'],
            'close': row['close'],
            'signal': row['signal']
        })
        if len(data) > 500:
            data.pop(0)
        time.sleep(1)

# Dashboard Configuration
app = Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id='live-graph'),
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
])

@app.callback(
    Output('live-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    df = pd.DataFrame(data)
    if df.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['close'], mode='lines', name='BTCUSDC Prices', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['price'], mode='lines', name='Estimated Prices', line=dict(color='purple')))
    buys = df[df['signal'] == 'buy']
    sells = df[df['signal'] == 'sell']
    fig.add_trace(go.Scatter(x=buys['timestamp'], y=buys['price'], mode='markers', marker=dict(color='green', symbol='triangle-up', size=10), name='Buy Signal'))
    fig.add_trace(go.Scatter(x=sells['timestamp'], y=sells['price'], mode='markers', marker=dict(color='red', symbol='triangle-down', size=10), name='Sell Signal'))
    fig.update_layout(
        title='Live BTC Price with Trade Signals',
        xaxis_title='Timestamp',
        yaxis_title='Price (USD)',
        template='plotly_dark'
    )
    return fig

# Start Data Generation
threading.Thread(target=update_data, daemon=True).start()

if __name__ == "__main__":
    app.run(debug=True)
