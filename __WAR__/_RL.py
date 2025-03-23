import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from datetime import timedelta
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/iliemoromete/Desktop/GLITCH-Production/__WAR__")
from __path__ import *

from _deltaT import *
period_extractor = DeltaTextractor(PATH_PIVOTS, PATH_PIVOT_PERIODS)
period_extractor.run()

# Load the data with timezone consistency and reset index
pivots_df = pd.read_csv(PATH_PIVOT_PERIODS, parse_dates=['buy_time', 'sell_time'])
news_df = pd.read_csv(PATH_NEWS_IMPACT, parse_dates=['news_headline_publication_timestamp', 'insignificance_timestamp'])
news_df['impact_normalized'] = news_df['impact'] / news_df['impact'].max()

# Ensure timestamps are timezone-naive for comparison
pivots_df['buy_time'] = pivots_df['buy_time'].dt.tz_localize(None)
pivots_df['sell_time'] = pivots_df['sell_time'].dt.tz_localize(None)
news_df['news_headline_publication_timestamp'] = news_df['news_headline_publication_timestamp'].dt.tz_localize(None)
news_df['insignificance_timestamp'] = news_df['insignificance_timestamp'].dt.tz_localize(None)

def find_closest_timestamp(news_time):
    """Find the closest pivot timestamp to a given news time."""
    all_pivot_times = pd.concat([pivots_df['buy_time'], pivots_df['sell_time']]).sort_values().reset_index(drop=True)
    news_time = pd.to_datetime(news_time)
    all_pivot_times = pd.to_datetime(all_pivot_times)
    closest_time = all_pivot_times.iloc[(all_pivot_times - news_time).abs().idxmin()]
    return closest_time

# Custom Reinforcement Learning Environment
class HarveySpecter(gym.Env):
    def __init__(self):
        super(HarveySpecter, self).__init__()

        # Load data before defining spaces
        self.load_data()

        # Define action space (predicting pivot time within 200-minute window)
        self.action_space = spaces.Box(low=np.array([0.0]), high=np.array([200.0]), dtype=np.float32)

        # Define observation space with 6 dimensions (new features added)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(6,), dtype=np.float32
        )
        self.current_index = 0
        self.state = None

    def load_data(self):
        """Load and preprocess data with additional price-related features."""
        self.historical_data = pd.read_csv(PATH_KLINES, parse_dates=['timestamp'])
        self.historical_data.sort_values(by='timestamp', inplace=True)

        # Add price behavior features with robust .fillna(0) handling
        self.historical_data['price_momentum'] = self.historical_data['close'].pct_change(5).fillna(0)
        self.historical_data['volatility'] = self.historical_data['close'].rolling(window=5).std().fillna(0)
        self.historical_data['price_gap'] = (self.historical_data['close'] - 
                                            self.historical_data['close'].shift(5)).fillna(0)
        # In load_data() method
        self.historical_data.dropna(subset=['price_momentum', 'volatility', 'price_gap'], inplace=True)

        # Ensure no NaN values remain
        assert not self.historical_data[['price_momentum', 'volatility', 'price_gap']].isnull().values.any(), \
            "Error: NaN values detected in price features!"
        print(f"Loaded {len(self.historical_data)} rows with price features.")

    def reset(self, seed=None, options=None):
        self.current_index = 0
        self.state = self._get_state()
        if np.isnan(self.state).any():
            print("❗ NaN detected in reset state")
            self.state = np.nan_to_num(self.state)
        return self.state, {}

    def _get_state(self):
        """Retrieve the current state based on news impact, pivot periods, and enhanced price behavior."""
        if self.current_index >= len(news_df):
            self.current_index = len(news_df) - 1 

        news_row = news_df.iloc[self.current_index]

        # Align historical data using merge_asof for correct timestamp matching
        closest_price_data = pd.merge_asof(
            news_df[['news_headline_publication_timestamp']],
            self.historical_data,
            left_on='news_headline_publication_timestamp',
            right_on='timestamp',
            direction='backward'
        ).iloc[self.current_index]

        # Enhanced NaN protection
        price_momentum = closest_price_data['price_momentum']
        volatility = closest_price_data['volatility']
        price_gap = closest_price_data['price_gap']

        # Pivot Periods
        closest_pivot_time = find_closest_timestamp(news_row['news_headline_publication_timestamp'])
        pivot_row = pivots_df[
            (pivots_df['buy_time'] == closest_pivot_time) | 
            (pivots_df['sell_time'] == closest_pivot_time)
        ]
        pivot_period = pivot_row['period_minutes'].values[0] if not pivot_row.empty else 1  # Avoid zero division

        # Feature calculations
        time_diff = abs((closest_pivot_time - news_row['news_headline_publication_timestamp']).total_seconds()) / 60
        impact = news_row['impact']
        normalized_period = pivot_period / 1440  # Normalize by 1 day (1440 mins)

        state = np.array([
            impact / 100 if np.isfinite(impact) else 0,
            time_diff / 1440 if np.isfinite(time_diff) else 0,
            normalized_period if np.isfinite(normalized_period) else 0,
            price_momentum,
            volatility,
            price_gap
        ], dtype=np.float32)
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

        if not np.isfinite(state).all():
            print(f"❗ Invalid state detected at index {self.current_index}: {state}")
            state = np.nan_to_num(state)  # Replace NaNs/Infs with zero

        return state

    def step(self, action):
        if self.current_index >= len(news_df):
            done = True
            truncated = False
            return self.state, 0.0, done, truncated, {}
        news_row = news_df.iloc[self.current_index]
        predicted_pivot_time = news_row['news_headline_publication_timestamp'] + timedelta(minutes=int(action[0]))
        closest_pivot_time = find_closest_timestamp(news_row['news_headline_publication_timestamp'])

        # Reward Logic: Assess trading performance based on actual price movement
        price_now = self.historical_data.iloc[self.current_index]['close']
        future_index = min(self.current_index + 5, len(self.historical_data) - 1)
        price_next = self.historical_data.iloc[future_index]['close']

        # Improved Reward Calculation
        if not np.isnan(price_now) and not np.isnan(price_next):
            if action > 100:   # 'Buy' Signal
                profit = price_next - price_now
            elif action < 50:  # 'Sell' Signal
                profit = price_now - price_next
            else:
                profit = 0     # Hold/Neutral Position
        else:
            profit = 0  # Default zero if data is unreliable
        reward = max(profit, -20)  # Cap max loss to prevent extreme penalties
        if np.isnan(reward):
            reward = 0.0

        if np.isnan(reward) or np.isnan(price_now) or np.isnan(price_next):
            print(f"❗ NaN detected at index {self.current_index}")
            print(f"price_now: {price_now}, price_next: {price_next}, reward: {reward}")
        self.current_index += 1
        done = self.current_index >= len(news_df)
        truncated = False
        return self._get_state(), reward, done, truncated, {}

    def render(self, mode='human'):
        print(f"State: {self.state}")

class HarveySpecterBacktest(HarveySpecter):
    def __init__(self, historical_data_path):
        super().__init__()
        self.historical_data = pd.read_csv(historical_data_path, parse_dates=['timestamp'])
        self.historical_data.sort_values(by='timestamp', inplace=True)

        # Initialize portfolio
        self.cash = 1000
        self.holdings = 0
        self.portfolio_values = []
        self.buy_signals = []
        self.sell_signals = []

    def backtest(self, model):
        with open("backtest_log.txt", "w") as log_file:
            for index, row in self.historical_data.iterrows():
                state = self._get_state()
                action, _ = model.predict(state)
                print(f"{index} - State: {state}, Action: {action}")
                log_file.write(f"{index} - State: {state}, Action: {action}\n")
                price = row['close']
                if action > 100 and self.cash > 0:
                    self.holdings = self.cash / price
                    self.cash = 0
                    self.buy_signals.append((row['timestamp'], price))  # Track buy signals
                elif action < 50 and self.holdings > 0:
                    self.cash = self.holdings * price
                    self.holdings = 0
                    self.sell_signals.append((row['timestamp'], price))  # Track sell signals
                portfolio_value = self.cash + (self.holdings * price)
                self.portfolio_values.append(portfolio_value)
                # Final results
                final_value = self.portfolio_values[-1]
                total_return = ((final_value - 1000) / 1000) * 100
        print(f"Final Portfolio Value: ${final_value:.2f}")
        print(f"Total Return: {total_return:.2f}%")

    def plot_results(self):
        """Plot both portfolio growth and buy/sell signals."""
        timestamps = self.historical_data['timestamp']

        # Plot 1: Price data with Buy/Sell signals
        plt.figure(figsize=(30, 16))
        plt.plot(timestamps, self.historical_data['close'], label='Price Data', color='blue')
        
        if self.buy_signals:
            buy_times, buy_prices = zip(*self.buy_signals)
            plt.scatter(buy_times, buy_prices, color='green', marker='^', label='Buy', s=100)

        if self.sell_signals:
            sell_times, sell_prices = zip(*self.sell_signals)
            plt.scatter(sell_times, sell_prices, color='red', marker='v', label='Sell', s=100)

        plt.title("Crypto Price Data with Buy/Sell Signals")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot 2: Portfolio Growth
        plt.figure(figsize=(15, 8))
        plt.plot(timestamps, self.portfolio_values, label='Portfolio Value', color='orange')
        plt.title("Portfolio Growth Over Time")
        plt.xlabel("Time")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True)
        plt.show()


class MikeRoss(gym.Env):
    def __init__(self, klines_path, future_trades_path):
        super(MikeRoss, self).__init__()
        self.klines_path = klines_path
        self.future_trades_path = future_trades_path
        self.load_data()

        self.buy_signals = []
        self.sell_signals = []

        self.action_space = spaces.Discrete(3)  # 0 = Hold, 1 = Buy, 2 = Sell
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

        self.cash = 1000  
        self.holdings = 0
        self.portfolio_values = []  
        self.current_index = 0
        self.state = None

    def load_data(self):
        self.historical_data = pd.read_csv(self.klines_path, parse_dates=['timestamp'])
        self.historical_data.sort_values(by='timestamp', inplace=True)
        self.future_data = pd.read_csv(self.future_trades_path, parse_dates=['timestamp'])
        self.future_data.sort_values(by='timestamp', inplace=True)

        self.historical_data['price_momentum'] = self.historical_data['close'].pct_change(5).fillna(0)
        self.historical_data['volatility'] = self.historical_data['close'].rolling(window=5).std().fillna(0)
        self.historical_data['price_gap'] = (self.historical_data['close'] - self.historical_data['close'].shift(5)).fillna(0)
        self.historical_data.dropna(inplace=True)

    def reset(self, seed=None, options=None):
        self.current_index = 0
        self.cash = 1000
        self.holdings = 0
        self.portfolio_values = []
        self.state = self._get_state()
        return self.state, {}

    def _get_state(self):
        closest_future_row = self.future_data.iloc[self.current_index]
        closest_historical_row = self.historical_data.iloc[self.current_index]

        state = np.array([
            closest_historical_row['price_momentum'],
            closest_historical_row['volatility'],
            closest_historical_row['price_gap'],
            self.cash / 1000,
            self.holdings,
            closest_future_row['price'] / 100000
        ], dtype=np.float32)

        return np.nan_to_num(state)

    def calculate_reward(self, predicted_price, actual_price):
        error = abs(predicted_price - actual_price) / actual_price
        if error < 0.001:
            return 10
        elif error < 0.005:
            return 5
        elif error < 0.01:
            return 2
        else:
            return -10 * error

    def step(self, action):
        if self.current_index >= len(self.future_data) - 1:
            done = True
            truncated = False
            return self.state, 0.0, done, truncated, {}

        actual_price = self.future_data.iloc[self.current_index]['price']

        # Reward Logic (simple)
        reward = 0
        if action == 1 and self.cash > 0:  # Buy signal
            self.holdings = self.cash / actual_price
            self.cash = 0
            self.buy_signals.append((self.future_data.iloc[self.current_index]['timestamp'], actual_price))
            reward = 10
        elif action == 2 and self.holdings > 0:  # Sell signal
            self.cash = self.holdings * actual_price
            self.holdings = 0
            self.sell_signals.append((self.future_data.iloc[self.current_index]['timestamp'], actual_price))
            reward = 10

        portfolio_value = self.cash + self.holdings * actual_price
        self.portfolio_values.append((self.future_data.iloc[self.current_index]['timestamp'], portfolio_value))

        self.current_index += 1
        self.state = self._get_state()
        done = self.current_index >= len(self.future_data)
        truncated = False

        return self.state, reward, done, truncated, {}

    def render(self, mode='human'):
        print(f"Step {self.current_index}: Portfolio Value = {self.portfolio_values[-1][1]:.2f}")

    def plot_results(self):
        df_future = pd.read_csv(self.future_trades_path, parse_dates=['timestamp'])
        df_klines = pd.read_csv(self.klines_path, parse_dates=['timestamp'])
        df_klines.dropna(subset=['timestamp'], inplace=True)

        print(f"NaN values in BTCUSDC_klines timestamps: {df_klines['timestamp'].isna().sum()}")
        print(f"NaN values in future_trades timestamps: {df_future['timestamp'].isna().sum()}")

        df_klines['timestamp'] = pd.to_datetime(df_klines['timestamp'])
        df_klines.sort_values('timestamp', inplace=True)
        df_future['timestamp'] = pd.to_datetime(df_future['timestamp'])
        df_future.sort_values('timestamp', inplace=True)

        merged_data = pd.merge_asof(
            df_klines[['timestamp', 'close']], 
            df_future[['timestamp', 'price', 'signal']], 
            on='timestamp',
            direction='backward'
        )

        merged_data.dropna(subset=['close', 'price'], inplace=True)

        plt.figure(figsize=(30, 16))
        plt.plot(merged_data['timestamp'], merged_data['price'], color='purple', label='Estimated Prices')
        plt.plot(merged_data['timestamp'], merged_data['close'], color='blue', label='Historical BTCUSDC Prices')

        buy_signals = merged_data[merged_data['signal'] == 'buy']
        sell_signals = merged_data[merged_data['signal'] == 'sell']

        if not buy_signals.empty:
            plt.scatter(buy_signals['timestamp'], buy_signals['price'], color='green', label='Buy Signal', marker='^')
        if not sell_signals.empty:
            plt.scatter(sell_signals['timestamp'], sell_signals['price'], color='red', label='Sell Signal', marker='v')

        plt.title("BTC Price with Trade Signals")
        plt.xlabel("Timestamp")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{PATH_TRADE}/plot_prices.png")
        plt.close()

        # Portfolio Growth
        portfolio_df = pd.DataFrame(self.portfolio_values, columns=['timestamp', 'portfolio_value'])

        plt.figure(figsize=(30, 16))
        plt.plot(portfolio_df['timestamp'], portfolio_df['portfolio_value'], color='orange', label='Portfolio Value')
        plt.title("Portfolio Value Over Time")
        plt.xlabel("Timestamp")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{PATH_TRADE}/plot_portfolio.png")
        plt.close()