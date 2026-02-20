
import vectorbt as vbt
import pandas as pd
import numpy as np
import logging
from data import loader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("vbt_trend")

def load_clean_data():
    df = loader.load_processed()
    df = df[df.index.year < 2025] # Exclude 2025
    return df['close']

def run_trend_strategy():
    close_m1 = load_clean_data()
    
    timeframes = ['5T', '15T', '1H', '4H']
    
    # MA Parameters
    fast_windows = [10, 20, 50]
    slow_windows = [50, 100, 200]
    
    for tf in timeframes:
        logger.info(f"--- Testing Timeframe: {tf} ---")
        close_tf = close_m1.resample(tf).last().dropna()
        
        # Run MAs
        # We need cartesian product of fast/slow.
        # But fast must be < slow.
        # VBT param_product=True will do all against all.
        # We can filter later or just ignore inverted (Fast > Slow is same as Slow < Fast but inverted signal?)
        
        fast_ma = vbt.MA.run(close_tf, window=fast_windows, param_product=True)
        slow_ma = vbt.MA.run(close_tf, window=slow_windows, param_product=True)
        
        # We need to align them.
        # They have different columns (window=10 vs window=50).
        # We can't easily broadcast two param_product objects against each other in one go 
        # unless indices match.
        # VBT Indicator Factory supports multiple inputs?
        
        # Simplest way: Loop
        for fast_w in fast_windows:
            for slow_w in slow_windows:
                if fast_w >= slow_w:
                    continue
                
                # Independent Runs
                fast = vbt.MA.run(close_tf, window=[fast_w])
                slow = vbt.MA.run(close_tf, window=[slow_w])
                
                # entries = fast.ma_crossed_above(slow)
                # exits = fast.ma_crossed_below(slow)
                
                entries = fast.ma > slow.ma
                exits = fast.ma < slow.ma
                # This is "Always In" Trend Following
                
                pf = vbt.Portfolio.from_signals(
                    close_tf,
                    entries=entries,
                    exits=exits, 
                    short_entries=exits,
                    short_exits=entries,
                    freq=tf,
                    init_cash=10000,
                    fees=0.00012,
                    slippage=0.00005
                )
                
                # Metrics
                total_return = pf.total_return()
                sharpe = pf.sharpe_ratio()
                trades = pf.trades.count()
                max_dd = pf.max_drawdown()
                
                if sharpe > 0.5:
                     logger.info(f"TF: {tf} | MA: {fast_w}/{slow_w} | Sharpe: {sharpe:.2f} | Return: {total_return:.2%} | Trades: {trades} | MaxDD: {max_dd:.2%}")

if __name__ == "__main__":
    run_trend_strategy()
