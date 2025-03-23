import pandas as pd
from _pivot import PivotButPurple  

class SignalUpdater:
    def __init__(self, estimated_bcktst_path, output_path, interval='1m', window_size=27, significance_threshold=0.0000000000001):
        self.estimated_bcktst_path = estimated_bcktst_path
        self.output_path = output_path
        self.interval = interval
        self.window_size = window_size
        self.significance_threshold = significance_threshold

    def update_signals(self):
        # Load predicted BTC data
        estimated_bcktst_df = pd.read_csv(self.estimated_bcktst_path, parse_dates=['timestamp'])

        # Run PivotButPurple directly on the estimated BTC data
        pivot_detector = PivotButPurple(
            estimated_bcktst_df,
            interval=self.interval,
            window_size=self.window_size,
            significance_threshold=self.significance_threshold,
            output_path=self.output_path
        )
        pivot_detector.run()
        print(f"✅ Future trade signals saved to {self.output_path}")
