
import vectorbt as vbt
import pandas as pd
import numpy as np
import logging
from data import loader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("vbt_runner")

def load_clean_data():
    logger.info("Loading full M1 data...")
    df = loader.load_processed()
    # Critical Step: Remove 2025 (The Anomalous Year)
    df = df[df.index.year < 2025]
    logger.info(f"Clean Data (2016-2024): {len(df)} bars")
    return df['close']

def run_vbt_test():
    close_m1 = load_clean_data()
    
    # Resample
    timeframes = ['5T', '15T', '1H', '4H']
    results = {}
    
    for tf in timeframes:
        logger.info(f"Resampling to {tf}...")
        close_tf = close_m1.resample(tf).last().dropna()
        
        # Simple Strategy: MA Crossover
        fast_ma = vbt.MA.run(close_tf, 10)
        slow_ma = vbt.MA.run(close_tf, 50)
        
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)
        
        # Portfolio
        pf = vbt.Portfolio.from_signals(
            close_tf, 
            entries, 
            exits, 
            freq=tf,
            init_cash=10000,
            fees=0.00012, # 1.2 pips spread equivalent commission
            slippage=0.00005 # 0.5 pips slippage
        )
        
        total_return = pf.total_return()
        sharpe = pf.sharpe_ratio()
        max_dd = pf.max_drawdown()
        
        logger.info(f"--- Results {tf} ---")
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"Sharpe: {sharpe:.2f}")
        logger.info(f"Max DD: {max_dd:.2%}")
        
        results[tf] = {
            'total_return': total_return,
            'sharpe': sharpe,
            'max_dd': max_dd
        }
        
    return results

if __name__ == "__main__":
    run_vbt_test()
