
import vectorbt as vbt
import pandas as pd
import numpy as np
import logging
from research.pivot_baselines import load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pivot_h4")

def run_h4_refinement():
    """
    Day 3/4: H4 Timeframe Refinement.
    H1 was too noisy (Baselines < 0.5 Sharpe on full dataset).
    Hypothesis: H4 filters noise and captures the "Tradeable Trend".
    
    Assets: NQ_F, BTC_USD
    Timeframe: 4H
    Strategies:
    1.  MA Cross (50/200, 20/50).
    2.  Slope Filter (Slow MA > 0).
    """
    logger.info("Starting H4 Timeframe Refinement...")
    
    for asset in ["NQ_F", "BTC_USD"]:
        # Resample to 4H
        df = load_data(asset)
        close_h1 = df['close'] 
        close_h4 = close_h1.resample('4H').last().dropna()
        
        logger.info(f"--- {asset} (H4 Bars: {len(close_h4)}) ---")
        
        # Test 1: Classic Golden Cross (50/200)
        fast = vbt.MA.run(close_h4, 50)
        slow = vbt.MA.run(close_h4, 200)
        
        entries = fast.ma_crossed_above(slow)
        exits = fast.ma_crossed_below(slow)
        
        pf_classic = vbt.Portfolio.from_signals(close_h4, entries, exits, freq='4h', fees=0.0006, slippage=0.0001)
        
        # Test 2: Faster Trend (20/50) + Slope Filter
        f2 = vbt.MA.run(close_h4, 20)
        s2 = vbt.MA.run(close_h4, 50)
        
        slope = s2.ma.diff()
        entries_2 = f2.ma_crossed_above(s2) & (slope > 0)
        exits_2 = f2.ma_crossed_below(s2)
        
        pf_fast = vbt.Portfolio.from_signals(close_h4, entries_2, exits_2, freq='4h', fees=0.0006, slippage=0.0001)
        
        logger.info(f"  [Classic 50/200] Sharpe: {pf_classic.sharpe_ratio():.2f} | Ret: {pf_classic.total_return():.2%}")
        logger.info(f"  [Fast 20/50 + Slope] Sharpe: {pf_fast.sharpe_ratio():.2f} | Ret: {pf_fast.total_return():.2%}")
        
        if pf_fast.sharpe_ratio() > 1.0:
            logger.info("  >> STRONG CANDIDATE FOUND <<")

if __name__ == "__main__":
    run_h4_refinement()
