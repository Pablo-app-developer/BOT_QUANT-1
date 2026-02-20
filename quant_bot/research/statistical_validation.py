
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from data import loader
from strategies.snapback_m5 import generate_signals

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("stat_valid")

def run_statistical_validation():
    """
    Phase 6.2: Statistical Validation.
    Methods:
    1. Monte Carlo (Shuffled Trades).
    2. Bootstrap (Resampled Trades).
    3. Year-by-Year Stability (Walk-Forward proxy).
    """
    logger.info("Loading data...")
    df = loader.load_processed()
    
    # Generate Trades (Theoretical)
    # We use the logic from audit to get triggers
    # Or just generate_signals output
    # Let's use specific logic to get Return per Trade
    
    close = df['close']
    open_ = df['open']
    
    # Re-calculate Triggers
    rets = close.pct_change().fillna(0)
    vol_60 = rets.rolling(60).std()
    vol_threshold = vol_60.quantile(0.66)
    is_high_vol = vol_60 > vol_threshold
    
    sma_fast = close.rolling(25).mean()
    sma_slow = close.rolling(100).mean()
    trend_up = sma_fast > sma_slow
    trend_down = sma_fast < sma_slow
    
    momentum = close.pct_change(15)
    mom_std = momentum.rolling(1000).std()
    threshold = 3.5 * mom_std
    
    long_triggers = (momentum < -threshold) & trend_up & is_high_vol
    short_triggers = (momentum > threshold) & trend_down & is_high_vol
    
    # Calculate Returns
    # We assume Limit Execution at Close[t] (Best case 90% fill)
    # Exit at Close[t+15]
    # Cost: Spread (1.0) + Commission (0.0). No Gap cost if Limit Close.
    # Risk: Validating the "Ideal" edge first.
    # If Ideal Edge fails MC, then Real Edge definitely fails.
    
    # Trade Returns
    # Short: Entry Close[t] -> Exit Close[t+15]
    s_idx = np.where(short_triggers)[0]
    s_idx = s_idx[s_idx < len(df) - 15]
    ret_short = (close.values[s_idx] - close.values[s_idx+15]) / close.values[s_idx]
    
    # Long: Entry Close[t] -> Exit Close[t+15]
    l_idx = np.where(long_triggers)[0]
    l_idx = l_idx[l_idx < len(df) - 15]
    ret_long = (close.values[l_idx+15] - close.values[l_idx]) / close.values[l_idx]
    
    all_rets = np.concatenate([ret_short, ret_long])
    
    # Subtract Costs (1 pip spread ~ 1 bps)
    # EURUSD 1 pip = 0.0001
    # 1 bps = 0.0001
    cost_bps = 0.0001 # 1 pip cost
    net_rets = all_rets - cost_bps
    
    logger.info(f"Total Trades: {len(net_rets)}")
    logger.info(f"Mean Net Return: {np.mean(net_rets)*10000:.2f} bps")
    logger.info(f"Win Rate: {np.mean(net_rets > 0):.1%}")
    
    # ── 1. MONTE CARLO (Shuffle) ──
    # Shuffle order of trades to destroy serial correlation and sequence luck
    # Does not destroy the edge itself, but tests Drawdown distribution
    
    n_sims = 1000
    initial_equity = 10000
    risk_per_trade = 0.01 # Fixed fractional? Or fixed lots?
    # Let's assume fixed money risk or simple sum of returns for Speed
    # Cumulative Sum of Returns * Equity
    
    max_dds = []
    final_equities = []
    
    logger.info(f"Running {n_sims} Monte Carlo simulations (Shuffle)...")
    
    for i in range(n_sims):
        shuffled = np.random.permutation(net_rets)
        # Equity Curve (compounding)
        # equity = 10000 * cumprod(1 + ret)
        curve = np.cumprod(1 + shuffled)
        
        peak = np.maximum.accumulate(curve)
        dd = (peak - curve) / peak
        max_dd = np.max(dd)
        
        max_dds.append(max_dd)
        final_equities.append(curve[-1])
        
    avg_dd = np.mean(max_dds)
    p95_dd = np.percentile(max_dds, 95)
    prob_ruin = np.mean(np.array(max_dds) > 0.20) # 20% DD limit
    
    logger.info(f"MC Avg MaxDD: {avg_dd:.1%}")
    logger.info(f"MC 95%ile MaxDD: {p95_dd:.1%}")
    logger.info(f"Prob Ruin (>20% DD): {prob_ruin:.1%}")
    
    # ── 2. YEARLY STABILITY ──
    logger.info("--- Yearly Breakdown ---")
    df['year'] = df.index.year
    years = df['year'].unique()
    
    yearly_sharp = []
    
    for y in sorted(years):
        # Slice DataFrame?
        # Faster to just use index mask
        mask_y = (df.index.year == y)
        # We need to map trade indices to years
        # s_idx are indices in df
        
        # Filter triggers in year y
        s_y = s_idx[df.index[s_idx].year == y]
        l_y = l_idx[df.index[l_idx].year == y]
        
        # Recalculate rets for year y
        rs = (close.values[s_y] - close.values[s_y+15]) / close.values[s_y]
        rl = (close.values[l_y+15] - close.values[l_y]) / close.values[l_y]
        
        y_rets = np.concatenate([rs, rl]) - cost_bps
        
        if len(y_rets) > 0:
            mean_y = np.mean(y_rets)
            sum_y = np.sum(y_rets)
            sharpe_y = mean_y / np.std(y_rets) * np.sqrt(len(y_rets)) # Rough annual sharpe proxy
            logger.info(f"Year {y}: Trades={len(y_rets)}, Sum={sum_y*10000:.0f} bps, Sharpe~={sharpe_y:.2f}")
            if sum_y < 0:
                logger.warning(f"Year {y} is LOSING!")
        else:
            logger.info(f"Year {y}: No trades")

if __name__ == "__main__":
    run_statistical_validation()
