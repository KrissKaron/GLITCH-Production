import asyncio
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta, datetime
import sys, os
import logging
from stable_baselines3 import PPO
from __path__ import *
logging.basicConfig(filename=f'{PATH_LOGS}{SE}scraper.log',
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.append(PATH_APIS)
sys.path.append(PATH_CLASS)
from _MDC import MDC
from _pivot import *
from _impact import *
from _amplitude import *
from _deltaT import *
from _RL import *
from _signal_updater import *
from _plottation_device import *
from _trade import *
# Importing API modules
from API_Sentiment import RedditSentimentScraper
from API_Newsapi import API_Newsapi
from API_Regulatory import API_Regulatory
from API_Whales import WhaleAlertCollector
from API_MacroEconomic import EconomicDataCollector
from API_MarketManipulation import WhaleTrackingCollector
from API_Mining import MiningDataCollector
from API_Binance import API_Binance

# Import API keys
from __keys__ import (
    API_REDDIT_CLIENTID, API_REDDIT_CLIENTSECRET, API_REDDIT_CLIENTUSERAGENT,
    API_NEWSAPI, API_CRYPTOPANIC, API_WHALES, API_POLYGON, API_SANTIMENT,
    API_COINWARZ, API_BINANCE_PUBLIC, API_BINANCE_PRIVATE
)

# Function to run synchronous API calls in an executor
def run_sync_task(task, *args, **kwargs):
    return task(*args, **kwargs)

async def fetch_reddit_sentiment():
    try:
        prompts = [
            "Bitcoin", "BTC", "BTC price", "Bitcoin price", "BTC adoption", "Bitcoin adoption",
            "BTC ta", "Bitcoin technical analysis", "Bitcoin mining", "BTC news", "Bitcoin news",
            "halving", "BTC halving", "Bitcoin halving", "BTC bull", "BTC bear", "Bitcoin bear",
            "Bitcoin bull"
        ]
        scraper = RedditSentimentScraper(API_REDDIT_CLIENTID, API_REDDIT_CLIENTSECRET, API_REDDIT_CLIENTUSERAGENT)
        return await asyncio.get_event_loop().run_in_executor(None, scraper.collect_and_save_sentiment_data, prompts, 100)
    except Exception as e:
        print(f"Error fetching Reddit sentiment: {e}")
        return None

async def fetch_whale_alerts():
    whale_alert = WhaleAlertCollector(API_WHALES)
    return await asyncio.get_event_loop().run_in_executor(None, whale_alert.collect_and_save_transactions, "btc", 500000)

async def fetch_regulatory_news():
    try:
        regulatory_news = API_Regulatory(API_CRYPTOPANIC)
        await asyncio.get_event_loop().run_in_executor(None, regulatory_news.collect_and_save_regulatory_news, 30)
        logging.info("Regulatory news data fetched and saved successfully.")
    except Exception as e:
        logging.error(f"Error fetching regulatory news: {e}")

async def fetch_general_news():
    try:
        news_api = API_Newsapi(API_NEWSAPI)
        await asyncio.get_event_loop().run_in_executor(None, news_api.collect_news_data, 30)
        logging.info("General news data fetched and saved successfully.")
    except Exception as e:
        logging.error(f"Error fetching general news: {e}")

async def fetch_mining_data():
    try:
        mining_data_collector = MiningDataCollector(api_key=API_COINWARZ)
        await asyncio.get_event_loop().run_in_executor(None, mining_data_collector.collect_and_save_profitability_data)
        logging.info("Mining data fetched and saved successfully.")
    except Exception as e:
        logging.error(f"Error fetching mining data: {e}")

async def fetch_on_chain_data():
    try:
        whale_tracking_collector = WhaleTrackingCollector(jwt_token=API_SANTIMENT)
        await asyncio.get_event_loop().run_in_executor(None, whale_tracking_collector.collect_and_save_on_chain_data, "bitcoin")
        logging.info("On-chain data fetched and saved successfully.")
    except Exception as e:
        logging.error(f"Error fetching on-chain data: {e}")

async def fetch_economic_news():
    try:
        economic_data_collector = EconomicDataCollector(api_key=API_POLYGON)
        await asyncio.get_event_loop().run_in_executor(None, economic_data_collector.collect_and_save_economic_news, 50)
        logging.info("Economic news data fetched and saved successfully.")
    except Exception as e:
        logging.error(f"Error fetching economic news: {e}")

async def fetch_binance_klines():
    try:
        file_path = PATH_KLINES

        # Determine dynamic start date based on existing data
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)

            # Ensure 'timestamp' is correctly parsed as datetime
            if 'timestamp' in existing_df.columns and not existing_df.empty:
                existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'], errors='coerce')

                # Drop any invalid timestamps (e.g., NaT values)
                existing_df.dropna(subset=['timestamp'], inplace=True)

                if not existing_df.empty:
                    latest_timestamp = existing_df['timestamp'].max()
                    start_date = int(latest_timestamp.timestamp() * 1000)
                else:
                    start_date = int((datetime.now() - timedelta(days=5)).timestamp() * 1000)
            else:
                start_date = int((datetime.now() - timedelta(days=5)).timestamp() * 1000)
        else:
            start_date = int((datetime.now() - timedelta(days=5)).timestamp() * 1000)

        end_date = int((datetime.now()).timestamp() * 1000)

        _api_binance_ = API_Binance(API_BINANCE_PUBLIC, API_BINANCE_PRIVATE)
        client, server_timestamp, acc_snapshot = _api_binance_.execute()
        print(f"Fetching Binance Klines from {datetime.utcfromtimestamp(start_date / 1000)} to {datetime.utcfromtimestamp(end_date / 1000)}")
        
        mdc = MDC(client, '1m', start_date, end_date)
        result = await asyncio.get_event_loop().run_in_executor(None, mdc.get_historical_data)

        if result is None or result.empty:
            logging.warning("No new Binance Klines data available.")
        else:
            logging.info("Binance Klines data fetched and saved successfully.")
    except Exception as e:
        logging.error(f"Error fetching Binance klines data: {e}")
        print(f"Error: {e}")

async def excavate_score_compact():
    extractor = CSVExcavator(PATH_ROOT)
    scorer = NewsScorer()

    # Extract data asynchronously with error handling
    general_news_df = await asyncio.get_event_loop().run_in_executor(None, extractor.extract_general_news)
    interest_rate_df = await asyncio.get_event_loop().run_in_executor(None, extractor.extract_interest_rate_news)
    regulatory_news_df = await asyncio.get_event_loop().run_in_executor(None, extractor.extract_regulatory_news)
    whale_transactions_df = await asyncio.get_event_loop().run_in_executor(None, extractor.extract_whale_transactions)

    # Ensure data is not None; replace with an empty DataFrame if necessary
    general_news_df = general_news_df if general_news_df is not None else pd.DataFrame(columns=['title', 'description', 'content', 'published_at'])
    interest_rate_df = interest_rate_df if interest_rate_df is not None else pd.DataFrame(columns=['title', 'description', 'published_at'])
    regulatory_news_df = regulatory_news_df if regulatory_news_df is not None else pd.DataFrame(columns=['title', 'description', 'published_at'])
    whale_transactions_df = whale_transactions_df if whale_transactions_df is not None else pd.DataFrame(columns=['datetime', 'value'])

    # Score the news asynchronously
    scored_general_news = await asyncio.get_event_loop().run_in_executor(None, scorer.score_general_news, general_news_df)
    scored_interest_rate = await asyncio.get_event_loop().run_in_executor(None, scorer.score_interest_rate_news, interest_rate_df)
    scored_regulatory_news = await asyncio.get_event_loop().run_in_executor(None, scorer.score_regulatory_news, regulatory_news_df)
    scored_whale_transactions = await asyncio.get_event_loop().run_in_executor(None, scorer.score_whale_transactions, whale_transactions_df)

    # Instantiate the compacter class AFTER scoring is done
    compacter = CSVCompacter(scored_general_news, scored_interest_rate, scored_regulatory_news, scored_whale_transactions)
    await asyncio.get_event_loop().run_in_executor(None, compacter.save_to_csv)
    print("Data successfully compacted and saved.")
    print("================================= 8 =================================\n\n")

async def run_pivot_detection(klines_file, output_path):
    klines_df = pd.read_csv(klines_file)
    pivot_detector = Pivot(klines_df, interval='1m', window_size=5, significance_threshold=0.001, output_path=output_path)
    await asyncio.get_event_loop().run_in_executor(None, pivot_detector.run)

async def run_deltaT_extraction(input_path, output_path):
    analyzer = DeltaTextractor(input_path, output_path)
    await asyncio.get_event_loop().run_in_executor(None, analyzer.run)

async def run_news_impact(input_file, output_file):
    impactor = Impactor(input_file, output_file)
    await asyncio.get_event_loop().run_in_executor(None, impactor.load_data)
    impact_df = await asyncio.get_event_loop().run_in_executor(None, impactor.calculate_impact)
    impact_df = impactor.remove_low_impact(impact_df)
    await asyncio.get_event_loop().run_in_executor(None, impactor.save_impact_to_csv, impact_df)
    print(impact_df.head())

# PURPLE WORKFLOW DESCRIPTION:
##############################################################################################################################
# 1. Model.zip
# 2. PIVOT_PERIODS_1m.csv & NEWS_IMPACT.csv - you get the buy and sell time from pivot periods, 
# and the publication timestamp and insignificance timestamp from news. You calculate 
# price_momentum, volatility and price_gap for closest_price_data. The model then uses RL to predict 
# next action and the new impact.
# 3. NEWS_PERIODS.csv - previous step's newly adjusted impact is saved here. 
# 4. bcktst.csv & NEWS_PERIODS.csv - you get the timestamp from the klines and the timestamp from news, 
# and you make sure they are identical. BTCNewsImpactEstimator will then generate the purple line equivalent 
# to the BTC kline close price (the blue line in the plot is historical, the purple is estimated. The purple line values
# for BTC are saved in estimated_prices.csv).
# 5. SignalUpdater will take the timestamp and price from estimated_prices.csv, and the signals from purple_signals.csv, 
# and combine the 2 files to obtain estimated_bcktst.csv. Whose timestamp and purple_signal will be used in trading.

# ======================== MODEL LOADING ========================
async def PURPLE_train_or_load_model():
    MODEL_PATH = f"{PATH_ROOT}{SE}RL_model.zip"
    env = HarveySpecter()

    if os.path.exists(MODEL_PATH):
        print("✅ RL model found. Loading model...")
        model = PPO("MlpPolicy", env, learning_rate=0.001, batch_size=512,
                    gamma=0.99, clip_range=0.2, ent_coef=0.01, verbose=1, device='cpu')
        model.load(MODEL_PATH.replace(".zip", ""))  # Stable Baselines loads without .zip
    else:
        print("🚨 RL model not found. Training new model...")
        model = PPO("MlpPolicy", env, learning_rate=0.001, batch_size=512,
                    gamma=0.99, clip_range=0.2, ent_coef=0.01, verbose=1, device='cpu')
        model.learn(total_timesteps=100000)
        model.save(MODEL_PATH.replace(".zip", ""))
        print("✅ Model trained and saved successfully.")
    return model

# ======================== DATA PROCESSING ========================
async def PURPLE_run_period_extraction(input_file, output_file):
    analyzer = DeltaTextractor(input_file, output_file)
    print("================================= 12 =================================\n\n")
    await asyncio.get_event_loop().run_in_executor(None, analyzer.run)

async def PURPLE_run_impact_analysis(input_file, output_file):
    impactor = Impactor(input_file, output_file)
    print("================================= 13 =================================\n\n")
    await asyncio.get_event_loop().run_in_executor(None, impactor.run)

async def PURPLE_adjust_news_impact(pivot_file, news_impact_file, model, output_file):
    pivot_df = pd.read_csv(pivot_file, parse_dates=['buy_time', 'sell_time'])
    news_df = pd.read_csv(news_impact_file, parse_dates=['news_headline_publication_timestamp', 'insignificance_timestamp'])
    pivot_df['buy_time']                            = pivot_df['buy_time'].dt.tz_localize(None)
    pivot_df['sell_time']                           = pivot_df['sell_time'].dt.tz_localize(None)
    news_df['news_headline_publication_timestamp']  = news_df['news_headline_publication_timestamp'].dt.tz_localize(None)
    news_df['insignificance_timestamp']             = news_df['insignificance_timestamp'].dt.tz_localize(None)

    adjusted_news = []
    for _, row in news_df.iterrows():
        closest_price_data = pd.merge_asof(
            news_df[['news_headline_publication_timestamp']],
            pivot_df[['buy_time', 'sell_time', 'period_minutes']],
            left_on='news_headline_publication_timestamp',
            right_on='buy_time',
            direction='backward'
        ).iloc[0]

        price_momentum = closest_price_data.get('price_momentum', 0)
        volatility = closest_price_data.get('volatility', 0)
        price_gap = closest_price_data.get('price_gap', 0)

        state = np.array([
            row['impact'] / 100,
            0.5,
            0.5,
            price_momentum,
            volatility,
            price_gap
        ], dtype=np.float32)

        action, _ = model.predict(state)
        new_impact = max(0, row['impact'] + action[0])

        decay_factor = 0.01
        new_period_minutes = int(np.log(10 / max(new_impact, 1e-6)) / -decay_factor)
        new_insignificance_timestamp = row['news_headline_publication_timestamp'] + timedelta(minutes=new_period_minutes)

        adjusted_news.append({
            'assessed_news_headline_id': row['assessed_news_headline_id'],
            'impact': new_impact,
            'news_headline_publication_timestamp': row['news_headline_publication_timestamp'],
            'insignificance_timestamp': new_insignificance_timestamp,
            'period_minutes': new_period_minutes
        })

    adjusted_df = pd.DataFrame(adjusted_news)
    adjusted_df.to_csv(output_file, index=False)
    print("✅ Adjusted news impact saved.")
    print("================================= 14 =================================\n\n")

# ======================== PLOTTING ========================
async def PURPLE_run_price_estimation(btc_data_file, news_data_file, damped_output):
    btc_data = pd.read_csv(btc_data_file, dtype=str)
    btc_data['timestamp'] = pd.to_datetime(btc_data['timestamp'], errors='coerce')
    news_data = pd.read_csv(news_data_file, dtype=str)
    news_data['timestamp'] = pd.to_datetime(news_data['news_headline_publication_timestamp'], errors='coerce')
    news_periods = pd.read_csv(news_data_file, dtype=str)
    news_periods['news_headline_publication_timestamp'] = pd.to_datetime(news_periods['news_headline_publication_timestamp'], errors='coerce')
    news_periods['impact'] = pd.to_numeric(news_periods['impact'], errors='coerce')
    news_periods['period_minutes'] = pd.to_numeric(news_periods['period_minutes'], errors='coerce')

    estimator = BTCNewsImpactEstimator()
    _, max_future_timestamp = estimator.extend_btc_data(btc_data, news_periods)
    estimator.plot_with_news_events(
        btc_data=btc_data, 
        news_data=news_data, 
        news_periods=news_periods,
        start_date='2025-03-14',
        end_date=max_future_timestamp,
        output_csv=damped_output
    )
    print("================================= 15 =================================\n\n")

# ======================== SIGNAL UPDATING ========================
async def PURPLE_update_signals(damped_file, output_file):
    updater = SignalUpdater(damped_file, output_file)
    await asyncio.get_event_loop().run_in_executor(None, updater.update_signals)
    print("================================= 16 =================================\n\n")

# ======================== TRADING THE FUTURE ========================
async def TRADE_future(future_trades_path, klines_path):
    trader = Trade(future_trades_path, klines_path)
    await asyncio.get_event_loop().run_in_executor(None, trader.simulate_trading)
    await asyncio.get_event_loop().run_in_executor(None, trader.plot_results)
    print("✅ Trading simulation and plots completed successfully.")
    print("================================= 17 =================================\n\n")

# ======================== REWARDING/PUNISHING PURPLE PIVOTS ========================
async def TRADE_MikeRoss(klines_path, future_trades_path):
    agent = MikeRoss(klines_path, future_trades_path)
    await asyncio.get_event_loop().run_in_executor(None, agent.plot_results)
    print("✅ Trading simulation complete.")