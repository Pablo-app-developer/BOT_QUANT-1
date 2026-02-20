
import pandas as pd
import numpy as np
import logging
from data import loader
from strategies.snapback_m5 import generate_signals

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("audit_edge")

def audit_edge():
    """
    Phase 6.1 Audit: Forensic analysis of the Edge.
    Checks:
    1. Gap Analysis (Open[t+1] vs Close[t]).
    2. Wick Analysis (High[t] vs Close[t]).
    3. Fill Probability Estimate.
    """
    logger.info("loading data for audit...")
    df = loader.load_processed()
    logger.info(f"Data loaded: {len(df)} bars")
    
    # Generate Signals (Ref: SnapBack M5 logic)
    # We need the RAW conditions to identify the "Trigger Bar"
    
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']
    
    # Re-calculate Entry Conditions locally for transparency
    rets = close.pct_change().fillna(0)
    vol_60 = rets.rolling(60).std()
    
    # Use global quantile for audit speed (avoid O(N*Window) rolling quantile)
    vol_threshold = vol_60.quantile(0.66)
    is_high_vol = vol_60 > vol_threshold
    
    sma_fast = close.rolling(25).mean()
    sma_slow = close.rolling(100).mean()
    trend_up = sma_fast > sma_slow
    trend_down = sma_fast < sma_slow
    
    momentum = close.pct_change(15)
    mom_std = momentum.rolling(1000).std()
    
    # Threshold 3.5 Sigma
    threshold = 3.5 * mom_std
    
    # Entry Signals (Trigger happens at Close[t])
    long_triggers = (momentum < -threshold) & trend_up & is_high_vol
    short_triggers = (momentum > threshold) & trend_down & is_high_vol
    
    logger.info(f"Triggers Found: Long={long_triggers.sum()}, Short={short_triggers.sum()}")
    
    # ── 1. GAP ANALYSIS ──
    # Gap = Open[t+1] - Close[t]
    # Cost for Short: Open[t+1] < Close[t] (We sell lower than trigger) -> Gap is Negative
    # Cost for Long: Open[t+1] > Close[t] (We buy higher than trigger) -> Gap is Positive
    
    # Get indices
    s_idx = np.where(short_triggers.values)[0]
    s_idx = s_idx[s_idx < len(df) - 1] # Valid next bars
    
    l_idx = np.where(long_triggers.values)[0]
    l_idx = l_idx[l_idx < len(df) - 1]
    
    # Calculate Gaps
    gap_short = open_.values[s_idx+1] - close.values[s_idx]
    gap_long = open_.values[l_idx+1] - close.values[l_idx]
    
    gap_short_pips = gap_short / 0.0001
    gap_long_pips = gap_long / 0.0001
    
    # Slippage (Cost is positive)
    # Short: We want Open >= Close. If Open < Close, we lost value.
    # Slip = Close - Open = -Gap. (If Gap is -5, Slip is +5 pips cost).
    slip_short = -gap_short_pips
    
    # Long: We want Open <= Close. If Open > Close, we lost value.
    # Slip = Open - Close = Gap. (If Gap is +5, Slip is +5 pips cost).
    slip_long = gap_long_pips
    
    avg_slip_short = np.mean(slip_short)
    avg_slip_long = np.mean(slip_long)
    total_avg_slip = (np.sum(slip_short) + np.sum(slip_long)) / (len(slip_short) + len(slip_long))
    
    logger.info("--- GAP ANALYSIS (Execution Friction) ---")
    logger.info(f"Avg Gap Cost Short: {avg_slip_short:.2f} pips")
    logger.info(f"Avg Gap Cost Long:  {avg_slip_long:.2f} pips")
    logger.info(f"TOTAL AVG GAP COST: {total_avg_slip:.2f} pips")
    
    # Distribution of Gaps
    # How often is Gap > 0? (Favorable for Short?)
    # Short Slip < 0 means Gap > 0 (Open > Close). We Sold higher!
    favorable_shorts = np.sum(slip_short < 0) / len(slip_short)
    favorable_longs = np.sum(slip_long < 0) / len(slip_long)
    logger.info(f"Favorable Gaps (Slippage < 0): Short={favorable_shorts:.1%}, Long={favorable_longs:.1%}")
    
    # ── 2. WICK ANALYSIS ──
    # Check if the "Trigger Close" was the High/Low of the bar.
    # If Close == High, the move was one-directional up to the last second.
    # If Close < High, there was already intra-bar rejection.
    # Strategy relies on "Overextension at Close".
    
    # For Shorts: Check (High[t] - Close[t])
    wick_short = high.values[s_idx] - close.values[s_idx]
    wick_short_pips = wick_short / 0.0001
    
    # For Longs: Check (Close[t] - Low[t])
    wick_long = close.values[l_idx] - low.values[l_idx]
    wick_long_pips = wick_long / 0.0001
    
    logger.info("--- WICK ANALYSIS (Intra-bar Reversion) ---")
    logger.info(f"Avg Upper Wick (Short Triggers): {np.mean(wick_short_pips):.2f} pips")
    logger.info(f"Avg Lower Wick (Long Triggers):  {np.mean(wick_long_pips):.2f} pips")
    
    # Interpretation:
    # Small Wick = Close is near High/Low = Strong Momentum at Close = Highly likely to Gap?
    # Large Wick = Rejection already started?
    
    # Correlation between Wick and Gap?
    # Do bars with NO wick (Strong Close) have Larger Gaps?
    corr_short = np.corrcoef(wick_short_pips, slip_short)[0, 1]
    logger.info(f"Correlation Wick vs FutureGap (Short): {corr_short:.2f}")
    
    # ── 3. FILL PROBABILITY ESTIMATE ──
    # If we used Limit Orders at Close[t].
    # Probability of fill depends on: Does price revisit Close[t] in the next bar?
    # Next bar is t+1.
    # Check if Next Low <= Close[t] <= Next High
    
    # Short: Sell Limit at Close[t].
    # Filled if Next High >= Close[t]. (Assuming we are passive at Close[t] price).
    # Actually, if Gap is Negative (Open < Close), price opened BELOW our Limit.
    # Price must rally back to Close[t] to fill us.
    # Condition: High[t+1] >= Close[t]
    
    next_highs_short = high.values[s_idx+1]
    fill_prob_short = np.mean(next_highs_short >= close.values[s_idx])
    
    # Long: Buy Limit at Close[t].
    # Filled if Next Low <= Close[t].
    next_lows_long = low.values[l_idx+1]
    fill_prob_long = np.mean(next_lows_long <= close.values[l_idx])
    
    logger.info("--- LIMIT ORDER FILL PROBABILITY (At Close Price) ---")
    logger.info(f"Short Fill Probability: {fill_prob_short:.1%}")
    logger.info(f"Long Fill Probability:  {fill_prob_long:.1%}")
    
    # ── 4. SPIKE DURATION ──
    # How many consecutive bars is the price > 3.5 Sigma?
    # This detects if "High Volatility" is a regime (clustering) or a single spike.
    
    # Not easy to calculate broadly, but let's check consecutive triggers.
    # If s_idx has t and t+1, it's a cluster.
    
    diff_s = np.diff(s_idx)
    clusters_s = np.sum(diff_s == 1)
    logger.info(f"Consecutive Short Triggers (Cluster): {clusters_s} pairs ({clusters_s/len(s_idx):.1%})")

if __name__ == "__main__":
    audit_edge()
