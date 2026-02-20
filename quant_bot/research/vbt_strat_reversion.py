
import vectorbt as vbt
import pandas as pd
import numpy as np
import logging
from data import loader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("vbt_reversion")

def load_clean_data():
    df = loader.load_processed()
    df = df[df.index.year < 2025] # Exclude 2025
    return df['close']

def run_reversion_strategy():
    close_m1 = load_clean_data()
    
    # Timeframes: Higher is better for Mean Rev?
    timeframes = ['1H', '4H']
    
    # RSI Parameters
    windows = [14]
    lower_thresholds = [20, 25, 30]
    upper_thresholds = [70, 75, 80]
    
    for tf in timeframes:
        logger.info(f"--- Testing Timeframe: {tf} ---")
        close_tf = close_m1.resample(tf).last().dropna()
        
        # RSI Indicator
        # vbt.RSI.run(close, window=windows)
        rsi = vbt.RSI.run(close_tf, window=windows, param_product=True)
        
        # We need to simulate combinations of Lower/Upper thresholds
        # VBT doesn't have a "Threshold Iterator" built-in for signals easily unless we custom build.
        # But we can iterate python-side or broadcast.
        
        # Let's iterate thresholds manually for clarity in logging
        for lower in lower_thresholds:
            for upper in upper_thresholds:
                # Long: RSI < Lower
                entries = rsi.rsi < lower
                # Short: RSI > Upper
                entries_short = rsi.rsi > upper
                
                # Exits?
                # Option A: Mean Reversion to 50?
                # Option B: Opposite Signal?
                # Let's try Exit at RSI 50 first (Classic Mean Rev)
                # exits_long = rsi.rsi > 50
                # exits_short = rsi.rsi < 50
                
                # Let's try Simple Reversal first (Long at Bottom, Short at Top)
                # entries = entries, short_entries = entries_short
                
                pf = vbt.Portfolio.from_signals(
                    close_tf,
                    entries=entries,
                    short_entries=entries_short,
                    freq=tf,
                    init_cash=10000,
                    fees=0.00012,
                    slippage=0.00005
                )
                
                total_return = pf.total_return()
                sharpe = pf.sharpe_ratio()
                trades = pf.trades.count()
                
                # Check metrics (Series if window broadcasted)
                # We expect scalar if window is single [14].
                # Wait, window is list [14]. rsi is VBT object.
                # If window len > 1, total_return is Series.
                # If window len = 1, total_return is Series (index=window).
                
                # Extract scalar if possible or print Series
                mean_sharpe = sharpe.mean()
                
                if mean_sharpe > 0.5:
                    logger.info(f"TF: {tf} | Thresh: {lower}/{upper} | Sharpe: {mean_sharpe:.2f} | Trades: {trades.sum()}")
                    # Print details
                    # print(f"Return: {total_return.mean():.2%}")

if __name__ == "__main__":
    run_reversion_strategy()
