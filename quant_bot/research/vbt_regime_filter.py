
import vectorbt as vbt
import pandas as pd
import numpy as np
import logging
from data import loader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("vbt_regime")

def load_clean_data():
    df = loader.load_processed()
    df = df[df.index.year < 2025] # Exclude 2025
    return df['close']

def run_regime_analysis():
    close_m1 = load_clean_data()
    
    # We focus on H1 for regimes
    tf = '1H'
    logger.info(f"--- Regime Analysis on {tf} ---")
    close = close_m1.resample(tf).last().dropna()
    
    # Define Regimes based on Volatility
    # Volatility = Rolling Std Dev of Returns (Window 50)
    vol = close.pct_change().rolling(50).std()
    
    # Quantiles
    low_vol = vol.quantile(0.33)
    high_vol = vol.quantile(0.66)
    
    # Masks
    is_low_vol = vol < low_vol
    is_high_vol = vol > high_vol
    is_normal_vol = (vol >= low_vol) & (vol <= high_vol)
    
    logger.info(f"Low Vol Threshold: {low_vol:.5f}")
    logger.info(f"High Vol Threshold: {high_vol:.5f}")
    
    # Strategy 1: Trend Following (MA Cross 20/50)
    fast = vbt.MA.run(close, 20)
    slow = vbt.MA.run(close, 50)
    entries_trend = fast.ma_crossed_above(slow)
    exits_trend = fast.ma_crossed_below(slow)
    
    # Strategy 2: Mean Reversion (RSI 14 < 30 / > 70)
    rsi = vbt.RSI.run(close, 14)
    entries_rev = rsi.rsi < 30
    entries_short_rev = rsi.rsi > 70
    # Simple reversal
    exits_rev = entries_short_rev
    exits_short_rev = entries_rev
    
    # Test Strategies Step-by-Step in Regimes
    regimes = {
        'All': None,
        'Low Vol': is_low_vol,
        'Normal Vol': is_normal_vol,
        'High Vol': is_high_vol
    }
    
    for regime_name, mask in regimes.items():
        logger.info(f"-> Testing Region: {regime_name}")
        
        # We enforce entry only if Regime Mask is True
        # VBT Portfolio from_signals doesn't have "mask" directly for filtering execution logic 
        # but we can AND logic with entries.
        
        # Test Trend
        if mask is None:
            t_entries = entries_trend
            t_exits = exits_trend
        else:
            t_entries = entries_trend & mask
            # Exits should ideally not be masked (we can exit in any regime), 
            # OR we force exit if regime changes? 
            # For now, let's just restrict ENTRIES to the regime.
            t_exits = exits_trend
            
        pf_trend = vbt.Portfolio.from_signals(
            close, entries=t_entries, exits=t_exits, 
            short_entries=t_exits, short_exits=t_entries,
            freq=tf, init_cash=10000, 
            fees=0.00012, slippage=0.00005
        )
        logger.info(f"  [Trend 20/50] Sharpe: {pf_trend.sharpe_ratio():.2f} | Returns: {pf_trend.total_return():.2%}")
        
        # Test Reversion
        if mask is None:
            r_entries = entries_rev
            r_entries_short = entries_short_rev
        else:
            r_entries = entries_rev & mask
            r_entries_short = entries_short_rev & mask
            
        pf_rev = vbt.Portfolio.from_signals(
            close, entries=r_entries, exits=r_entries_short,
            short_entries=r_entries_short, short_exits=r_entries,
            freq=tf, init_cash=10000,
            fees=0.00012, slippage=0.00005
        )
        logger.info(f"  [Reversion RSI] Sharpe: {pf_rev.sharpe_ratio():.2f} | Returns: {pf_rev.total_return():.2%}")

if __name__ == "__main__":
    run_regime_analysis()
