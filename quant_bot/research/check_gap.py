
import pandas as pd
import numpy as np
import logging
from data import loader
from strategies.snapback_m5 import generate_signals

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gap_check")

def check_gaps():
    df = loader.load_processed()
    logger.info(f"Data loaded: {len(df)}")
    
    signals = generate_signals(df, sigma=3.5, hold_period=15)
    
    # Identify entry bars
    # Signal changes from 0 to 1/ -1
    # Actually signals from generate_signals are already sparse (0 unless entry).
    # But wait, I changed generate_signals to extend the hold!
    # I need the ORIGINAL entries, not the extended position.
    
    # I will verify snapback_m5 logic again or just re-calculate entries here.
    # To be safe, I'll recalculate entries locally.
    
    close = df['close']
    open_ = df['open']
    
    # Re-calculate logic
    lookback=15
    rets = close.pct_change().fillna(0)
    vol_60 = rets.rolling(60).std()
    vol_threshold = vol_60.rolling(10_000, min_periods=1000).quantile(0.66)
    is_high_vol = vol_60 > vol_threshold
    
    sma_fast = close.rolling(25).mean()
    sma_slow = close.rolling(100).mean()
    trend_up = sma_fast > sma_slow
    trend_down = sma_fast < sma_slow
    
    momentum = close.pct_change(lookback)
    mom_std = momentum.rolling(1000).std()
    threshold_dynamic = 3.5 * mom_std
    
    long_entries = (momentum < -threshold_dynamic) & trend_up & is_high_vol
    short_entries = (momentum > threshold_dynamic) & trend_down & is_high_vol
    
    # ── Analyze Shorts ──
    # We trigger at close[t]. We enter at open[t+1].
    # Gap = open[t+1] - close[t]
    # For Short, we want to Sell High. If Gap < 0, we sell Lower (Bad).
    
    # Shift entries to match execution time (t+1)
    # execution_mask[t] corresponds to signal at t-1
    # But checking gap at t+1 relative to t
    
    idx_shorts = short_entries[short_entries].index
    # We want open[t+1] vs close[t] for these t
    
    # Get numeric indices
    # We use numpy for speed
    
    s_idx = np.where(short_entries.values)[0]
    # Ensure t+1 is valid
    s_idx = s_idx[s_idx < len(df) - 1]
    
    opens = open_.values
    closes = close.values
    
    # Gap = Open[t+1] - Close[t]
    gaps_short = opens[s_idx+1] - closes[s_idx]
    
    # Gap in Pips
    gaps_short_pips = gaps_short / 0.0001
    
    avg_gap_short = np.mean(gaps_short_pips)
    logger.info(f"Shorts ({len(s_idx)}): Avg Gap = {avg_gap_short:.2f} pips")
    
    # ── Analyze Longs ──
    # Trigger close[t]. Enter open[t+1].
    # We want Buy Low. 
    # Gap = Open[t+1] - Close[t]
    # If Gap > 0, we buy Higher (Bad).
    
    l_idx = np.where(long_entries.values)[0]
    l_idx = l_idx[l_idx < len(df) - 1]
    
    gaps_long = opens[l_idx+1] - closes[l_idx]
    gaps_long_pips = gaps_long / 0.0001
    
    avg_gap_long = np.mean(gaps_long_pips)
    logger.info(f"Longs ({len(l_idx)}): Avg Gap = {avg_gap_long:.2f} pips")
    
    # Total Slippage due to Gap (Bad Direction)
    # Short: Gap < 0 is Bad (Slippage = -Gap if Gap<0 else 0?)
    # No, effectively we enter at Open.
    # Theoretical Entry: Close[t]. Real Entry: Open[t+1].
    # Slippage = Entry - Ideal
    # Short Ideal: Sell at Close[t]. Real: Sell at Open[t+1].
    # Slip = Close[t] - Open[t+1] = -Gap.
    # If Gap is -2 pips (Open < Close), Slip is +2 pips (Cost).
    
    # Long Ideal: Buy at Close[t]. Real: Buy at Open[t+1].
    # Slip = Open[t+1] - Close[t] = Gap.
    # If Gap is +2 pips, Slip is +2 pips (Cost).
    
    slip_short = -gaps_short_pips
    slip_long = gaps_long_pips
    
    avg_slip_short = np.mean(slip_short)
    avg_slip_long = np.mean(slip_long)
    
    logger.info(f"Avg 'Natural Slippage' Short: {avg_slip_short:.2f} pips (Positive = Cost)")
    logger.info(f"Avg 'Natural Slippage' Long:  {avg_slip_long:.2f} pips (Positive = Cost)")
    
    total_natural_slippage = (np.sum(slip_short) + np.sum(slip_long)) / (len(s_idx) + len(l_idx))
    logger.info(f"TOTAL AVG NATURAL SLIPPAGE: {total_natural_slippage:.2f} pips")

if __name__ == "__main__":
    check_gaps()
