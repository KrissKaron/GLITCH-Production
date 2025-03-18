import pandas as pd

class SignalUpdater:
    def __init__(self, pivots_path, estimated_signals_path, output_path):
        self.pivots_path = pivots_path
        self.estimated_signals_path = estimated_signals_path
        self.output_path = output_path

    def update_signals(self):
        pivots_df = pd.read_csv(self.pivots_path, parse_dates=['timestamp'])
        estimated_signals_df = pd.read_csv(self.estimated_signals_path, parse_dates=['timestamp'])
        pivots_df['timestamp'] = pd.to_datetime(pivots_df['timestamp'])
        estimated_signals_df['timestamp'] = pd.to_datetime(estimated_signals_df['timestamp'])

        merged_signals = pd.merge(pivots_df, estimated_signals_df[['timestamp', 'signal']],
                                    on='timestamp', how='left', suffixes=('_original', '_purple'))
        merged_signals.rename(columns={'signal_original': 'original_signal', 'signal_purple': 'purple_signal'}, inplace=True)
        merged_signals['purple_signal'].fillna('hold', inplace=True)
        merged_signals.to_csv(self.output_path, index=False)
        print(f"New file '{self.output_path}' successfully created.")