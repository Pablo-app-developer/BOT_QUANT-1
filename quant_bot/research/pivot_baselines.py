
import vectorbt as vbt
import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pivot_baseline")

DATA_DIR = "quant_bot/data/pivot_ready"
ASSETS = ["BTC_USD", "ETH_USD", "NQ_F"]

def load_data(asset):
    path = f"{DATA_DIR}/{asset}_H1.parquet"
    df = pd.read_parquet(path)
    return df

def run_baselines():
    """
    Day 2: Minimum Baselines (Control).
    Run 3 types of strategies on TRAIN set only:
    1. Trend (MA Cross)
    2. Mean Reversion (RSI)
    3. Breakout (BB)
    """
    logger.info("Day 2: Running Baselines on TRAIN set...")
    
    results = []
    
    for asset in ASSETS:
        df = load_data(asset)
        # Filter Train
        train_df = df[df['split'] == 'train']
        close = train_df['close']
        
        logger.info(f"--- Asset: {asset} (Train Bars: {len(close)}) ---")
        
        # 1. Trend: MA Cross (Fast=10-50, Slow=50-200)
        fast_windows = [10, 20, 50]
        slow_windows = [50, 100, 200]
        
        # We use strict fees
        fees = 0.0006 # Crypto fees ~0.06% taker? For NQ ~0.01%, but let's be conservative.
        slippage = 0.0001
        
        best_sharpe = -999
        best_strat = ""
        
        # Trend Loop
        for f in fast_windows:
            for s in slow_windows:
                if f >= s: continue
                fast = vbt.MA.run(close, f)
                slow = vbt.MA.run(close, s)
                entries = fast.ma_crossed_above(slow)
                exits = fast.ma_crossed_below(slow)
                pf = vbt.Portfolio.from_signals(close, entries, exits, freq='1h', init_cash=10000, fees=fees, slippage=slippage)
                sh = pf.sharpe_ratio()
                if sh > best_sharpe:
                    best_sharpe = sh
                    best_strat = f"Trend MA({f},{s})"
        
        results.append({'Asset': asset, 'Type': 'Trend', 'Best Config': best_strat, 'Sharpe (Train)': best_sharpe})
        
        # 2. Mean Reversion: RSI (14) < 30 / > 70
        rsi = vbt.RSI.run(close, 14)
        entries = rsi.rsi < 30
        exits = rsi.rsi > 70
        pf = vbt.Portfolio.from_signals(close, entries, exits, short_entries=exits, short_exits=entries, freq='1h', init_cash=10000, fees=fees, slippage=slippage)
        sh = pf.sharpe_ratio()
        results.append({'Asset': asset, 'Type': 'MeanRev', 'Best Config': 'RSI(14, 30/70)', 'Sharpe (Train)': sh})

        # 3. Breakout: BB (20, 2.0)
        bb = vbt.BBANDS.run(close, 20, 2.0)
        entries = close > bb.upper
        exits = close < bb.lower
        pf = vbt.Portfolio.from_signals(close, entries, exits, short_entries=exits, short_exits=entries, freq='1h', init_cash=10000, fees=fees, slippage=slippage)
        sh = pf.sharpe_ratio()
        results.append({'Asset': asset, 'Type': 'Breakout', 'Best Config': 'BB(20, 2.0)', 'Sharpe (Train)': sh})
        
    # Report
    res_df = pd.DataFrame(results)
    logger.info("\n" + res_df.to_string())
    
    # Save Report
    res_df.to_csv("quant_bot/research/pivot_day2_baselines.csv")

if __name__ == "__main__":
    run_baselines()
