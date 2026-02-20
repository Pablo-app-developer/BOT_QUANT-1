
import pandas as pd
import numpy as np
import logging
from data import loader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("robustness")

def robustness_stress_test():
    """
    Phase 6.3: Parametric Destruction (Stress Test).
    Goal: Check if the edge exists in 'Normal Years' (2016-2024) under ANY parameter combination.
    If it fails everywhere, the strategy is purely an artifact of 2025.
    
    Params to Stress:
    - Sigma: [2.5, 3.0, 3.5, 4.0]
    - Holding: [5, 10, 15, 30]
    """
    logger.info("Loading data...")
    df = loader.load_processed()
    
    # Exclude 2025 (The Anomaly)
    df_clean = df[df.index.year != 2025].copy()
    logger.info(f"Data (Ex-2025): {len(df_clean)} bars. Validating Normal Regime.")
    
    close = df_clean['close']
    
    # Pre-calculate Indicators
    # Volatility
    vol_60 = close.pct_change().rolling(60).std()
    vol_threshold = vol_60.quantile(0.66)
    is_high_vol = vol_60 > vol_threshold
    
    # Trend
    sma_fast = close.rolling(25).mean()
    sma_slow = close.rolling(100).mean()
    trend_up = sma_fast > sma_slow
    trend_down = sma_fast < sma_slow
    
    # Momentum (Base 15m for signal generation)
    # We could stress Lookback too, but let's stick to Sigma/Hold first.
    momentum = close.pct_change(15)
    mom_std = momentum.rolling(1000).std()
    
    # ── Heatmap Loop ──
    sigmas = [2.5, 3.0, 3.5, 4.0, 4.5]
    holdings = [5, 10, 15, 30, 60]
    
    logger.info(f"{'Sigma':<6} | {'Hold':<6} | {'Trades':<8} | {'Net Bps':<10} | {'Win%':<6}")
    logger.info("-" * 50)
    
    valid_configs = 0
    
    for sigma in sigmas:
        threshold = sigma * mom_std
        
        # Signals
        long_t = (momentum < -threshold) & trend_up & is_high_vol
        short_t = (momentum > threshold) & trend_down & is_high_vol
        
        idx_l = np.where(long_t)[0]
        idx_s = np.where(short_t)[0]
        
        for hold in holdings:
            # Calculate Returns
            # Valid indices
            il = idx_l[idx_l < len(df_clean) - hold]
            is_ = idx_s[idx_s < len(df_clean) - hold]
            
            if len(il) + len(is_) < 50:
                continue
                
            # Log Returns
            rl = np.log(close.values[il+hold] / close.values[il])
            rs = np.log(close.values[is_] / close.values[is_+hold])
            
            all_rets = np.concatenate([rl, rs])
            
            # Net of Cost (1 pip spread ~ 1 bps)
            net_rets_bps = (all_rets * 10000) - 1.0
            
            mean_net = np.mean(net_rets_bps)
            win_rate = np.mean(net_rets_bps > 0)
            n_trades = len(all_rets)
            
            logger.info(f"{sigma:<6.1f} | {hold:<6} | {n_trades:<8} | {mean_net:>10.2f} | {win_rate:.1%}")
            
            if mean_net > 0.5: # Min exploitable edge
                valid_configs += 1
                
    if valid_configs == 0:
        logger.warning("CRITICAL: No parameter combination survives in Normal Years (2016-2024).")
    else:
        logger.info(f"Found {valid_configs} potentially valid configurations.")

if __name__ == "__main__":
    robustness_stress_test()
