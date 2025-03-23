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

# Data Loading (Separate Copies for Safety)
future_trades_df = pd.read_csv(PATH_FUTURE_TRADES).copy()
klines_df = pd.read_csv(PATH_KLINES).copy()

# Ensure timestamps are correctly formatted
future_trades_df['timestamp'] = pd.to_datetime(future_trades_df['timestamp'])
klines_df['timestamp'] = pd.to_datetime(klines_df['timestamp'])
klines_df.dropna(subset=['timestamp'], inplace=True)

# Filter BTCUSDC data to match `future_trades_df` start point
earliest_timestamp = future_trades_df['timestamp'].min()
klines_df = klines_df[klines_df['timestamp'] >= earliest_timestamp]

# Store separate in-memory copies for live updates
dashboard_future_trades = future_trades_df.copy()
dashboard_klines = klines_df.copy()

data = []  # Data storage
profits = []  # Profit tracking

# Data Synchronization and Storage (Initial Load)
merged_df = pd.merge_asof(
    future_trades_df.sort_values('timestamp'),
    klines_df.sort_values('timestamp'),
    on='timestamp',
    tolerance=pd.Timedelta('1min'),      # Prevents distant timestamps from merging
    direction='forward'                 # Ensures forward-matching timestamps
).dropna()

profit = 1000  # Initialize profit
last_trade_price = None

# Clear old data before adding new entries
data.clear()
profits.clear()

for _, row in merged_df.iterrows():
    data.append({
        'timestamp': row['timestamp'],
        'price': row['price'],
        'close': row['close'],
        'signal': row['signal']
    })
    if row['signal'] == 'buy':
        last_trade_price = row['price']
    elif row['signal'] == 'sell' and last_trade_price is not None:
        profit += row['price'] - last_trade_price
        profits.append({'timestamp': row['timestamp'], 'profit': profit})

# Live Data Update
def update_data():
    global data, profits, profit, last_trade_price
    while True:
        new_data = pd.merge_asof(
            dashboard_future_trades.sort_values('timestamp'),
            dashboard_klines.sort_values('timestamp'),
            on='timestamp',
            tolerance=pd.Timedelta('1min'),
            direction='forward'
        ).iloc[-1]  

        # Prevent duplicate timestamps
        if data and data[-1]['timestamp'] == new_data['timestamp']:
            time.sleep(1)
            continue

        data.append({
            'timestamp': new_data['timestamp'],
            'price': new_data['price'],
            'close': new_data['close'],
            'signal': new_data['signal']
        })

        if new_data['signal'] == 'buy':
            last_trade_price = new_data['price']
        elif new_data['signal'] == 'sell' and last_trade_price is not None:
            profit += new_data['price'] - last_trade_price
            profits.append({'timestamp': new_data['timestamp'], 'profit': profit})

        if len(data) > 500:
            data.pop(0)
        if len(profits) > 500:
            profits.pop(0)

        time.sleep(1)

# Dashboard Configuration
app = Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id='live-graph'),
    dcc.Graph(id='profit-graph'),
    dcc.Graph(id='price-24h-graph'),  # Added 24-hour plot
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
])

@app.callback(
    [
        Output('live-graph', 'figure'),
        Output('profit-graph', 'figure'),
        Output('price-24h-graph', 'figure')  # Added new output
    ],
    [
        Input('interval-component', 'n_intervals'),
        Input('live-graph', 'relayoutData')
    ]
)
def update_graph(n, relayout_data):
    df = pd.DataFrame(data)
    profit_df = pd.DataFrame(profits)
    if 'timestamp' not in profit_df.columns or profit_df.empty:
        profit_df = pd.DataFrame({'timestamp': [], 'profit': []})  # Safe fallback

    # Price Graph
    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(x=df['timestamp'], y=df['close'], mode='lines', name='BTCUSDC Prices', line=dict(color='blue')))
    price_fig.add_trace(go.Scatter(x=df['timestamp'], y=df['price'], mode='lines', name='Estimated Prices', line=dict(color='purple')))

    buys = df[df['signal'] == 'buy']
    sells = df[df['signal'] == 'sell']

    price_fig.add_trace(go.Scatter(x=buys['timestamp'], y=buys['price'], mode='markers', marker=dict(color='green', symbol='triangle-up', size=10), name='Buy Signal'))
    price_fig.add_trace(go.Scatter(x=sells['timestamp'], y=sells['price'], mode='markers', marker=dict(color='red', symbol='triangle-down', size=10), name='Sell Signal'))

    price_fig.update_layout(
        title='Live BTC Price with Trade Signals',
        xaxis_title='Timestamp',
        yaxis_title='Price (USD)',
        template='plotly_dark'
    )

    # Restore zoom/pan if available
    if relayout_data and 'xaxis.range' in relayout_data:
        price_fig.update_layout(xaxis=dict(range=relayout_data['xaxis.range']))

    # Profit Graph
    profit_fig = go.Figure()
    profit_fig.add_trace(go.Scatter(x=profit_df['timestamp'], y=profit_df['profit'], mode='lines', name='Profit', line=dict(color='green')))
    profit_fig.update_layout(
        title='Cumulative Profit Over Time',
        xaxis_title='Timestamp',
        yaxis_title='Profit (USD)',
        template='plotly_dark'
    )

    # Filter data for the last 24 hours
    last_24h_timestamp = df['timestamp'].max() - pd.Timedelta('1d')
    df_24h = df[df['timestamp'] >= last_24h_timestamp]

    # 24-hour Price Graph
    price_24h_fig = go.Figure()
    price_24h_fig.add_trace(go.Scatter(x=df_24h['timestamp'], y=df_24h['close'], mode='lines', name='BTCUSDC Prices (Last 24h)', line=dict(color='cyan')))
    price_24h_fig.add_trace(go.Scatter(x=df_24h['timestamp'], y=df_24h['price'], mode='lines', name='Estimated Prices (Last 24h)', line=dict(color='magenta')))
    # Add Buy/Sell Signals to 24-hour Graph
    buys_24h = df_24h[df_24h['signal'] == 'buy']
    sells_24h = df_24h[df_24h['signal'] == 'sell']
    price_24h_fig.add_trace(go.Scatter(x=buys_24h['timestamp'], y=buys_24h['price'], mode='markers', marker=dict(color='green', symbol='triangle-up', size=10), name='Buy Signal'))
    price_24h_fig.add_trace(go.Scatter(x=sells_24h['timestamp'], y=sells_24h['price'], mode='markers', marker=dict(color='red', symbol='triangle-down', size=10), name='Sell Signal'))
    price_24h_fig.update_layout(
        title='BTC Price in the Last 24 Hours',
        xaxis_title='Timestamp',
        yaxis_title='Price (USD)',
        template='plotly_dark'
    )
    
    return price_fig, profit_fig, price_24h_fig

threading.Thread(target=update_data, daemon=True).start()
def run_async_main():
    asyncio.run(main())
threading.Thread(target=run_async_main, daemon=True).start()
if __name__ == "__main__":
    app.run(debug=True)
