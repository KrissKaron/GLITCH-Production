import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

class BTCNewsPlotter:
    def __init__(self, bcktst_path, news_path):
        self.bcktst_df = pd.read_csv(bcktst_path, parse_dates=['timestamp'])
        self.news_df = pd.read_csv(news_path, parse_dates=['news_headline_publication_timestamp'])
        self.news_df['news_headline_publication_timestamp'] = self.news_df['news_headline_publication_timestamp'].dt.floor('min')

    def plot(self, start_date=None, end_date=None):
        # Filter BTC price data
        if start_date:
            self.bcktst_df = self.bcktst_df[self.bcktst_df['timestamp'] >= pd.to_datetime(start_date)]
        if end_date:
            self.bcktst_df = self.bcktst_df[self.bcktst_df['timestamp'] <= pd.to_datetime(end_date)]

        # Filter news timestamps to match the filtered BTC data range
        filtered_news_df = self.news_df[(self.news_df['news_headline_publication_timestamp'] >= self.bcktst_df['timestamp'].min()) &
                                        (self.news_df['news_headline_publication_timestamp'] <= self.bcktst_df['timestamp'].max())]

        # Plotting
        plt.figure(figsize=(30, 16))
        plt.plot(self.bcktst_df['timestamp'], self.bcktst_df['close'], label='Closing Price', color='blue')

        # Plot filtered news timestamps as purple dashed lines
        for news_time in filtered_news_df['news_headline_publication_timestamp']:
            plt.axvline(x=news_time, color='purple', linestyle='--', alpha=0.5)
        plt.title(f"Bitcoin Price with News Event Timestamps ({start_date or 'Start'} to {end_date or 'End'})")
        plt.xlabel("Time")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True)
        plt.show()

class BTCNewsPlotterHourly:
    def __init__(self, btc_file, news_file):
        self.btc_data = pd.read_csv(btc_file, parse_dates=['timestamp'])
        self.news_data = pd.read_csv(news_file, parse_dates=['news_headline_publication_timestamp'])

        # Adjust news timestamps to nearest minute
        self.news_data['news_headline_publication_timestamp'] = self.news_data[
            'news_headline_publication_timestamp'].dt.floor('min')

    def plot_data(self, start_date=None, end_date=None, start_hour=0, end_hour=23):
        # Filter BTC data by date and hour range
        filtered_btc = self.btc_data[
            (self.btc_data['timestamp'] >= pd.to_datetime(start_date)) &
            (self.btc_data['timestamp'] <= pd.to_datetime(end_date)) &
            (self.btc_data['timestamp'].dt.hour >= start_hour) &
            (self.btc_data['timestamp'].dt.hour <= end_hour)
        ]

        # Filter news data similarly
        filtered_news = self.news_data[
            (self.news_data['news_headline_publication_timestamp'] >= pd.to_datetime(start_date)) &
            (self.news_data['news_headline_publication_timestamp'] <= pd.to_datetime(end_date)) &
            (self.news_data['news_headline_publication_timestamp'].dt.hour >= start_hour) &
            (self.news_data['news_headline_publication_timestamp'].dt.hour <= end_hour)
        ]

        # Align news points with BTC closing prices
        news_close_prices = filtered_news.merge(filtered_btc[['timestamp', 'close']],
                                                left_on='news_headline_publication_timestamp',
                                                right_on='timestamp',
                                                how='inner')['close']

        # Plotting
        plt.figure(figsize=(30, 16))
        plt.plot(filtered_btc['timestamp'], filtered_btc['close'], label='Closing Price', color='blue')

        plt.scatter(filtered_news['news_headline_publication_timestamp'],
                    news_close_prices,
                    color='purple', label='News Event', s=50)  # Larger dots for visibility

        plt.title("Bitcoin Price with News Event Timestamps (Dots on Price Line)")
        plt.xlabel("Time")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True)
        plt.show()

class BTCNewsOscillatorPlotter:
    def __init__(self):
        self.max_future_minutes = 600  # Maximum number of future minutes to project

    def damped_oscillation(self, t, t0, amplitude, gamma, omega):
        return amplitude * np.exp(-gamma * (t - t0)) * np.cos(omega * (t - t0))

    def calculate_decay_point(self, amplitude, gamma, threshold=0.5):
        insignificance_threshold = amplitude * threshold
        decay_time = -np.log(insignificance_threshold / amplitude) / gamma
        return int(min(decay_time, self.max_future_minutes))  # Cap the decay point

    def extend_btc_data(self, btc_data, future_decay_points):
        """Extend BTC data to account for future timestamps from calculated decay points."""
        latest_timestamp = btc_data['timestamp'].max()
        max_future_point = min(max(future_decay_points), self.max_future_minutes)  # Cap future extension
        future_timestamps = pd.date_range(start=latest_timestamp, 
                                          periods=max_future_point + 1, 
                                          freq='1min')
        future_df = pd.DataFrame({'timestamp': future_timestamps, 'close': np.nan})
        return pd.concat([btc_data, future_df], ignore_index=True)

    def plot_with_news_events(self, btc_data, news_data, news_periods, start_date=None, end_date=None, start_hour=0, end_hour=23):
        btc_data['timestamp'] = pd.to_datetime(btc_data['timestamp'])
        btc_data['close'] = pd.to_numeric(btc_data['close'], errors='coerce')
        news_data['news_headline_publication_timestamp'] = pd.to_datetime(news_data['news_headline_publication_timestamp'])
        news_periods['news_headline_publication_timestamp'] = pd.to_datetime(news_periods['news_headline_publication_timestamp'])

        # Filter BTC data by date and time
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

        mask = (btc_data['timestamp'].dt.date >= start_date) & (btc_data['timestamp'].dt.date <= end_date)
        filtered_btc = btc_data[mask].reset_index(drop=True)
        filtered_btc.dropna(subset=['close', 'timestamp'], inplace=True)

        # Filter news data
        mask_news = (news_data['news_headline_publication_timestamp'].dt.date >= start_date) & (
            news_data['news_headline_publication_timestamp'].dt.date <= end_date)
        filtered_news = news_data[mask_news]

        # Determine future decay points
        future_decay_points = []

        plt.figure(figsize=(30, 16))
        plt.plot(filtered_btc['timestamp'], filtered_btc['close'], label='BTC Closing Price', color='blue')

        decay_point_plotted = False  # Control legend for decay points

        # Overlay harmonic oscillations with dynamic parameters
        for idx, news_time in filtered_news['news_headline_publication_timestamp'].items():
            closest_price_idx = (filtered_btc['timestamp'] - news_time).abs().idxmin()
            price_at_news = filtered_btc.loc[closest_price_idx, 'close']

            # Get corresponding impact and period values
            news_info = news_periods[news_periods['news_headline_publication_timestamp'] == news_time]

            if not news_info.empty:
                impact = news_info.iloc[0]['impact']
                period_minutes = news_info.iloc[0]['period_minutes']

                # Improved Parameter Mapping
                amplitude = impact * np.random.uniform(1.5, 4.5)  
                gamma = 1 / max(period_minutes ** 1.2, 5)
                omega = 2 * np.pi / max(period_minutes * np.random.uniform(0.8, 1.5), 3)
                phase_shift = np.random.uniform(0, 2 * np.pi)

                # Decay Point Calculation
                decay_point_idx = closest_price_idx + self.calculate_decay_point(amplitude, gamma)
                future_decay_points.append(decay_point_idx)

                # Extend BTC data to include the forecasted decay point
                if decay_point_idx >= len(filtered_btc):
                    future_timestamp = filtered_btc['timestamp'].max() + timedelta(minutes=decay_point_idx - len(filtered_btc) + 1)
                    
                    # ✅ Plot one decay point in the legend, subsequent points without label
                    if not decay_point_plotted:
                        plt.scatter(future_timestamp, price_at_news,
                                    color='red', marker='x', label='Decay Point (Signal)')
                        decay_point_plotted = True
                    else:
                        plt.scatter(future_timestamp, price_at_news,
                                    color='red', marker='x', label=None)

        # Extend BTC data for plotting the purple line
        extended_btc = self.extend_btc_data(filtered_btc, future_decay_points)

        # 🔹 Resize combined_impact to match extended BTC data size
        combined_impact = np.zeros(len(extended_btc))

        for idx, news_time in filtered_news['news_headline_publication_timestamp'].items():
            closest_price_idx = (extended_btc['timestamp'] - news_time).abs().idxmin()

            news_info = news_periods[news_periods['news_headline_publication_timestamp'] == news_time]
            if not news_info.empty:
                impact = news_info.iloc[0]['impact']
                period_minutes = news_info.iloc[0]['period_minutes']

                amplitude = impact * np.random.uniform(1.5, 4.5)
                gamma = 1 / max(period_minutes ** 1.2, 5)
                omega = 2 * np.pi / max(period_minutes * np.random.uniform(0.8, 1.5), 3)
                phase_shift = np.random.uniform(0, 2 * np.pi)

                end_index = min(closest_price_idx + 150, len(extended_btc) - 1)

                osc_time = np.arange(0, end_index - closest_price_idx)
                osc_values = self.damped_oscillation(osc_time, phase_shift, amplitude, gamma, omega)

                combined_impact[closest_price_idx:end_index] += osc_values

        # Purple Oscillator Line
        plt.plot(extended_btc['timestamp'], combined_impact[:len(extended_btc)] + extended_btc['close'].fillna(method='ffill'),
                 color='purple', linewidth=2, label='Estimated Oscillator (Purple Line)')

        plt.title('Bitcoin Price with Damped Harmonic Oscillations at News Events')
        plt.xlabel('Time')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()


    def plot_each_oscillator(self, btc_data, news_data, news_periods, start_date=None, end_date=None, start_hour=0, end_hour=23):
        btc_data['timestamp'] = pd.to_datetime(btc_data['timestamp'])
        btc_data['close'] = pd.to_numeric(btc_data['close'], errors='coerce')
        news_data['news_headline_publication_timestamp'] = pd.to_datetime(news_data['news_headline_publication_timestamp'])
        news_periods['news_headline_publication_timestamp'] = pd.to_datetime(news_periods['news_headline_publication_timestamp'])

        # Filter BTC data by date and time
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

        mask = (btc_data['timestamp'].dt.date >= start_date) & (btc_data['timestamp'].dt.date <= end_date)
        filtered_btc = btc_data[mask].reset_index(drop=True)
        filtered_btc.dropna(subset=['close', 'timestamp'], inplace=True)

        # Filter news data
        mask_news = (news_data['news_headline_publication_timestamp'].dt.date >= start_date) & (
            news_data['news_headline_publication_timestamp'].dt.date <= end_date)
        filtered_news = news_data[mask_news]

        plt.figure(figsize=(30, 16))
        plt.plot(filtered_btc['timestamp'], filtered_btc['close'], label='BTC Closing Price', color='blue')

        decay_point_plotted = False  # Control legend for decay points

        # Overlay individual green harmonic oscillations
        for idx, news_time in filtered_news['news_headline_publication_timestamp'].items():
            closest_price_idx = (filtered_btc['timestamp'] - news_time).abs().idxmin()
            price_at_news = filtered_btc.loc[closest_price_idx, 'close']

            # Get corresponding impact and period values
            news_info = news_periods[news_periods['news_headline_publication_timestamp'] == news_time]

            if not news_info.empty:
                impact = news_info.iloc[0]['impact']
                period_minutes = news_info.iloc[0]['period_minutes']

                # Improved Parameter Mapping
                amplitude = impact * np.random.uniform(1.5, 4.5)  
                gamma = 1 / max(period_minutes ** 1.2, 5)
                omega = 2 * np.pi / max(period_minutes * np.random.uniform(0.8, 1.5), 3)
                phase_shift = np.random.uniform(0, 2 * np.pi)

                # Decay Point Calculation
                decay_point_idx = closest_price_idx + self.calculate_decay_point(amplitude, gamma)

                # Individual Green Oscillation Line
                end_index = min(closest_price_idx + 150, len(filtered_btc) - 1)
                osc_time = np.arange(0, end_index - closest_price_idx)
                osc_values = self.damped_oscillation(osc_time, phase_shift, amplitude, gamma, omega) + price_at_news

                plt.plot(filtered_btc['timestamp'].iloc[closest_price_idx:end_index], osc_values, color='green', alpha=0.5)

                # Decay Point Plotting
                if not decay_point_plotted:
                    plt.scatter(filtered_btc['timestamp'].iloc[decay_point_idx], filtered_btc['close'].iloc[decay_point_idx],
                                color='red', marker='x', label='Decay Point (Signal)')
                    decay_point_plotted = True
                else:
                    plt.scatter(filtered_btc['timestamp'].iloc[decay_point_idx], filtered_btc['close'].iloc[decay_point_idx],
                                color='red', marker='x', label=None)

        plt.title('Bitcoin Price with Damped Harmonic Oscillations at News Events')
        plt.xlabel('Time')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()



class BTCNewsImpactEstimator:
    def __init__(self):
        pass

    def damped_oscillation(self, t, phase_shift, amplitude, gamma, omega):
        """Damped harmonic oscillator function."""
        return amplitude * np.exp(-gamma * t) * np.cos(omega * t + phase_shift)

    def calculate_decay_point(self, amplitude, gamma, threshold=0.5):
        """Calculate the timestamp where oscillations decay to negligible impact."""
        insignificance_threshold = amplitude * threshold
        decay_time = -np.log(insignificance_threshold / amplitude) / gamma
        return int(decay_time)

    def extend_btc_data(self, btc_data, news_periods):
        """Extend BTC data to cover future timestamps with dummy values."""
        future_timestamps = []
        for _, row in news_periods.iterrows():
            news_time = row['news_headline_publication_timestamp']
            period_minutes = row['period_minutes']
            future_timestamps.append(news_time + pd.Timedelta(minutes=period_minutes))
        max_future_timestamp = max(future_timestamps)

        # Create a DataFrame for the missing future timestamps
        future_data = pd.DataFrame({
            'timestamp': pd.date_range(start=btc_data['timestamp'].max(), 
                                    end=max_future_timestamp, 
                                    freq='1T')  # 1-minute intervals
        })
        # Fill dummy prices by propagating the last known price
        future_data['close'] = btc_data['close'].iloc[-1]
        extended_btc_data = pd.concat([btc_data, future_data]).reset_index(drop=True)
        return extended_btc_data, max_future_timestamp


    def plot_with_news_events(self, btc_data, news_data, news_periods, start_date=None, end_date=None, start_hour=0, end_hour=23, output_csv='estimated_prices.csv'):
        """Plots BTC closing prices, creates combined impact line, and saves the estimated prices as a CSV."""
        btc_data['timestamp'] = pd.to_datetime(btc_data['timestamp'])
        btc_data['close'] = pd.to_numeric(btc_data['close'], errors='coerce')
        news_data['news_headline_publication_timestamp'] = pd.to_datetime(news_data['news_headline_publication_timestamp'])
        news_periods['news_headline_publication_timestamp'] = pd.to_datetime(news_periods['news_headline_publication_timestamp'])

        # Extend BTC data to match projected oscillation range
        btc_data, max_future_timestamp = self.extend_btc_data(btc_data, news_periods)

        # Filter BTC data by date and time
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date = max_future_timestamp.date() if end_date is None else pd.to_datetime(end_date).date()

        mask = (btc_data['timestamp'].dt.date >= start_date) & (btc_data['timestamp'].dt.date <= end_date)
        filtered_btc = btc_data[mask].reset_index(drop=True)
        filtered_btc.dropna(subset=['close', 'timestamp'], inplace=True)

        # Filter news data
        mask_news = (news_data['news_headline_publication_timestamp'].dt.date >= start_date) & \
                    (news_data['news_headline_publication_timestamp'].dt.date <= end_date)
        filtered_news = news_data[mask_news]

        combined_impact = np.zeros(len(filtered_btc))

        #plt.figure(figsize=(20, 12))
        #plt.plot(filtered_btc['timestamp'], filtered_btc['close'], label='BTC Closing Price', color='blue')

        # Overlay harmonic oscillations with dynamic parameters
        for idx, news_time in filtered_news['news_headline_publication_timestamp'].items():
            closest_price_idx = (filtered_btc['timestamp'] - news_time).abs().idxmin()
            price_at_news = filtered_btc.loc[closest_price_idx, 'close']

            # Get corresponding impact and period values
            news_info = news_periods[news_periods['news_headline_publication_timestamp'] == news_time]

            if not news_info.empty:
                impact = news_info.iloc[0]['impact']
                period_minutes = news_info.iloc[0]['period_minutes']

                # Improved Parameter Mapping
                amplitude = impact * np.random.uniform(1.0, 4.5)  
                gamma = 1 / max(period_minutes ** 1.2, 5)        
                omega = 2 * np.pi / max(period_minutes * np.random.uniform(0.8, 1.5), 3)  
                phase_shift = np.random.uniform(0, 2 * np.pi)  

                # Oscillator range control
                end_index = min(closest_price_idx + 150, len(filtered_btc) - 1)

                # Generate oscillator time range
                osc_time = np.arange(0, end_index - closest_price_idx)
                osc_values = self.damped_oscillation(osc_time, phase_shift, amplitude, gamma, omega)

                combined_impact[closest_price_idx:end_index] += osc_values

        # Generate and plot the combined impact line
        combined_line = filtered_btc['close'] + combined_impact
        plt.plot(filtered_btc['timestamp'], combined_line, color='purple', alpha=0.7, label='Combined Impact (Estimated Price)')

        # Save estimated prices to CSV
        estimated_prices_df = pd.DataFrame({
            'timestamp': filtered_btc['timestamp'],
            'price': combined_line
        })

        estimated_prices_df.to_csv(output_csv, index=False)
        print(f"Estimated prices saved to '{output_csv}'.")

        #plt.title('Bitcoin Price with Damped Harmonic Oscillations at News Events')
        #plt.xlabel('Time')
        #plt.ylabel('Price (USD)')
        #plt.legend()
        #plt.grid(True)
        #plt.show()
        return filtered_btc, combined_impact


