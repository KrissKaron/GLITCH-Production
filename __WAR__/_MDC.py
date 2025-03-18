import pandas as pd
import os
from __path__ import *

class MDC:
    def __init__(self, binance_client, interval, start_time, end_time, symbol='BTCUSDC'):
        self.symbol = symbol
        self.interval = interval
        self.binance_client = binance_client
        self.start_time = start_time
        self.end_time = end_time
    
    def get_historical_data(self):
        klines = self.binance_client.get_historical_klines_generator(
            symbol=self.symbol,
            interval=self.interval,
            start_str=str(self.start_time),
            end_str=str(self.end_time)
        )
        
        # Convert the generator to a list and check if it's empty
        klines = list(klines)
        if not klines:
            print("No data fetched from Binance API.")
            return None
        
        print("Data fetched:", klines[:5])  # Print the first 5 rows to verify data

        # Convert data to DataFrame
        new_df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms') 
        file_path = PATH_KLINES

        # Check if the file exists to append new data
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            # Ensure timestamp is in correct datetime format
            existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'], errors='coerce')
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], errors='coerce')
            # Combine and remove duplicates based on timestamp
            combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['timestamp'], keep='first')
            combined_df.to_csv(file_path, index=False)
            print("New historical data appended successfully.")
        else:
            # Save new data if the file doesn't exist
            new_df.to_csv(file_path, index=False)
            print("Historical data saved successfully.")
        return new_df