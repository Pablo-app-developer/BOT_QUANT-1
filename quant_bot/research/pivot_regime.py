
import vectorbt as vbt
import pandas as pd
import numpy as np
import logging
from research.pivot_baselines import load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pivot_regime")

def run_regime_analysis():
    """
    Day 3: Regime Analysis on Nasdaq (NQ) and BTC.
    Goal: Check if the baseline Trend Edge improves with specific filters.
    
    1. Nasdaq (NQ): Session Filter.
       - Logic: Trends are strongest during US Cash Session (13:30 - 20:00 UTC).
       - (9:30 AM - 4:00 PM ET)
    2. Bitcoin (BTC): Volatility Filter.
       - Logic: Trends are safer when Volatility is increasing? Or Low?
       - Let's test ADX Strength Filter.
    """
    logger.info("Starting Day 3: Regime Analysis (NQ & BTC)...")
    
    # --- NASDAQ SESSION FILTER ---
    df_nq = load_data("NQ_F")
    close_nq = df_nq['close']
    
    # Define US Session (UTC)
    # 9:30 ET is ~14:30 UTC (Standard) or 13:30 (DST). 
    # Let's approximate 14:00 - 21:00 UTC for simplicity or use index hour.
    
    # VBT 50/200 Trend
    fast = vbt.MA.run(close_nq, 50)
    slow = vbt.MA.run(close_nq, 200)
    entries_nq = fast.ma_crossed_above(slow)
    exits_nq = fast.ma_crossed_below(slow)
    # Always In (Baseline)
    pf_base = vbt.Portfolio.from_signals(close_nq, entries_nq, exits_nq, short_entries=exits_nq, short_exits=entries_nq, freq='1h', fees=0.0001, slippage=0.0001)
    
    # Session Filter: Only take NEW entries during US Session.
    # Holding overnight is fine, but new signals only in liquidity.
    # Hour 14 to 20.
    hours = close_nq.index.hour
    us_session = (hours >= 14) & (hours <= 20)
    
    entries_filtered = entries_nq & us_session
    # Short entries also filtered?
    short_entries_filtered = exits_nq & us_session
    
    # Exits: Can exit anytime? Or only session?
    # Trend following usually holds. We just don't ENTER in Asia chop.
    # So we use standard exits.
    
    pf_filt = vbt.Portfolio.from_signals(
        close_nq, 
        entries=entries_filtered, 
        exits=exits_nq, 
        short_entries=short_entries_filtered, 
        short_exits=entries_filtered, 
        freq='1h', fees=0.0001, slippage=0.0001
    )
    
    logger.info(f"NQ Baseline Sharpe: {pf_base.sharpe_ratio():.2f}")
    logger.info(f"NQ Session Filter (14-20h UTC) Sharpe: {pf_filt.sharpe_ratio():.2f}")
    
    # --- BTC ADX FILTER ---
    df_btc = load_data("BTC_USD")
    close_btc = df_btc['close']
    high_btc = df_btc['high']
    low_btc = df_btc['low']
    
    # VBT 50/100 Trend
    fast_b = vbt.MA.run(close_btc, 50)
    slow_b = vbt.MA.run(close_btc, 100)
    entries_btc = fast_b.ma_crossed_above(slow_b)
    exits_btc = fast_b.ma_crossed_below(slow_b)
    
    # Baseline
    pf_btc_base = vbt.Portfolio.from_signals(close_btc, entries_btc, exits_btc, short_entries=exits_btc, short_exits=entries_btc, freq='1h', fees=0.0006, slippage=0.0001)
    
    # Filter: ADX > 25 (Strong Trend)
    # VBT doesn't have ADX easily? Use ATR as proxy for Vol?
    # Or simple Slope Filter: Slope(SlowMA) > 0 for Long?
    
    # Slope Filter
    slow_slope = slow_b.ma.diff()
    
    # Only Long if Slow MA is pointing UP (slope > 0)
    # Only Short if Slow MA is pointing DOWN (slope < 0)
    # This acts as a regime filter.
    
    entries_btc_filt = entries_btc & (slow_slope > 0)
    short_entries_btc_filt = exits_btc & (slow_slope < 0)
    
    pf_btc_filt = vbt.Portfolio.from_signals(
        close_btc,
        entries=entries_btc_filt,
        exits=exits_btc, # Exit on cross regardless
        short_entries=short_entries_btc_filt,
        short_exits=entries_btc_filt,
        freq='1h', fees=0.0006, slippage=0.0001
    )
    
    logger.info(f"BTC Baseline Sharpe: {pf_btc_base.sharpe_ratio():.2f}")
    logger.info(f"BTC Slope Filter (Trend Confirmation) Sharpe: {pf_btc_filt.sharpe_ratio():.2f}")

if __name__ == "__main__":
    run_regime_analysis()
