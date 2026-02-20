
import pandas as pd
import numpy as np
import logging
from data import loader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("audit_2025")

def audit_2025():
    """
    Forensic analysis of Year 2025.
    Why does it yield +16,000 bps while other years lose money?
    Hypothesis:
    1. Data artifacts (price spikes, bad ticks).
    2. Overlap between EURUSD2025.zip and EURUSD1.csv?
    3. Extreme volatility real event?
    """
    logger.info("Loading data...")
    df = loader.load_processed()
    
    df['year'] = df.index.year
    df_25 = df[df['year'] == 2025].copy()
    
    logger.info(f"2025 Data: {len(df_25)} bars")
    
    # Check Price Jumps
    close = df_25['close']
    high = df_25['high']
    low = df_25['low']
    
    # Gap Analysis in 2025
    open_ = df_25['open']
    gaps = open_.shift(-1) - close
    gaps_pips = gaps / 0.0001
    
    logger.info(f"2025 Max Gap: {gaps_pips.max():.2f} pips")
    logger.info(f"2025 Min Gap: {gaps_pips.min():.2f} pips")
    logger.info(f"2025 Avg Gap Abs: {gaps_pips.abs().mean():.2f} pips")
    
    # Check for "Zero Volatility" periods or "Flat lines"
    # Return = 0
    rets = close.pct_change().fillna(0)
    zeros = (rets == 0).sum()
    logger.info(f"2025 Zero Return Bars: {zeros} ({zeros/len(df_25):.1%})")
    
    # Check for "Bad Ticks" (High < Low)
    bad_ticks = (high < low).sum()
    logger.info(f"2025 Bad Ticks (H < L): {bad_ticks}")
    
    # Check Returns Distribution
    logger.info(f"2025 Max Single Bar Return: {rets.max():.4%}")
    logger.info(f"2025 Min Single Bar Return: {rets.min():.4%}")
    
    # Check Strategy Trades in 2025
    # Recalculate logic for 2025 subset? 
    # Or just slice triggers? (Need full data for indicators)
    
    # We need to run logic on FULL data then slice 2025 trades.
    # We already know from previous run that 2025 was profitable.
    # Let's see WHERE the profit comes from.
    # Are there a few trades with +1000 pips?
    
    # Logic Re-calc
    full_close = df['close']
    
    momentum = full_close.pct_change(15)
    mom_std = momentum.rolling(1000).std()
    threshold = 3.5 * mom_std
    
    sma_fast = full_close.rolling(25).mean()
    sma_slow = full_close.rolling(100).mean()
    trend_up = sma_fast > sma_slow
    trend_down = sma_fast < sma_slow
    
    vol_60 = full_close.pct_change().rolling(60).std()
    vol_threshold = vol_60.quantile(0.66)
    is_high_vol = vol_60 > vol_threshold
    
    long_t = (momentum < -threshold) & trend_up & is_high_vol
    short_t = (momentum > threshold) & trend_down & is_high_vol
    
    # Filter for 2025
    mask_25 = (df.index.year == 2025)
    # Indices in full df
    idx_l = np.where(long_t & mask_25)[0]
    idx_s = np.where(short_t & mask_25)[0]
    
    # Calculate returns
    # Long: Close[t+15] - Close[t]
    # Short: Close[t] - Close[t+15]
    
    # Be careful with bounds
    idx_l = idx_l[idx_l < len(df) - 15]
    idx_s = idx_s[idx_s < len(df) - 15]
    
    rl = (full_close.values[idx_l+15] - full_close.values[idx_l])/full_close.values[idx_l]
    rs = (full_close.values[idx_s] - full_close.values[idx_s+15])/full_close.values[idx_s]
    
    all_rets_25 = np.concatenate([rl, rs])
    all_rets_25_bps = all_rets_25 * 10000
    
    logger.info(f"2025 Trades: {len(all_rets_25)}")
    logger.info(f"2025 Max Trade Return: {all_rets_25_bps.max():.2f} bps")
    logger.info(f"2025 Min Trade Return: {all_rets_25_bps.min():.2f} bps")
    logger.info(f"2025 Mean Trade Return: {all_rets_25_bps.mean():.2f} bps")
    
    # Identify the outliers
    # Anomalous trades > 100 bps (100 pips in M1/M5 reversion?)
    outliers = all_rets_25_bps[np.abs(all_rets_25_bps) > 100]
    logger.info(f"2025 Outliers (>100 bps): {len(outliers)}")
    if len(outliers) > 0:
        logger.info(f"Outlier Values: {outliers}")

if __name__ == "__main__":
    audit_2025()
