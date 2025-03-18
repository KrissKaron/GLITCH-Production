from __path__ import *
import sys
sys.path.append(f'{PATH_WAR}')
sys.path.append(f'{PATH_CLASS}')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from stable_baselines3 import PPO
import asyncio
from async_methods import *
from _pivot import *
from _deltaT import *
from _impact import *
from _RL import *
from _amplitude import *
from _signal_updater import *
from _plottation_device import *

async def main():
    model = await PURPLE_train_or_load_model()
    while True:
        tasks = [
            fetch_general_news(),                                                   #1
            fetch_regulatory_news(),                                                #2
            fetch_whale_alerts(),                                                   #3
            fetch_economic_news(),                                                  #4
            fetch_on_chain_data(),                                                  #5
            fetch_mining_data(),                                                    #6
            fetch_binance_klines(),                                                 #7
            excavate_score_compact(),                                               #8
            run_pivot_detection(PATH_KLINES),                                       #9
            # P U R P L E
            PURPLE_run_period_extraction(PATH_PIVOTS_1M, PATH_PIVOT_PERIODS),                                           #10
            PURPLE_run_impact_analysis(PATH_NEWS, PATH_NEWS_IMPACT),                                                    #11
            PURPLE_adjust_news_impact(PATH_PIVOT_PERIODS, PATH_NEWS_IMPACT, model, PATH_NEWS_PERIODS),                  #12
            PURPLE_run_price_estimation(PATH_KLINES, PATH_NEWS_PERIODS, PATH_DAMPED),                                   #13
            PURPLE_update_signals(PATH_DAMPED, PATH_PIVOTS_1M, PATH_PURPLE_SIGNALS, PATH_ESTIMATED_BCKTST),             #14
            # Semi-live testing
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Task {idx+1} failed: {result}")
            else:
                print(f"Task {idx+1} completed successfully.")
        print("Waiting for next run...")
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())