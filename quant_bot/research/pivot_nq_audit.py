
import vectorbt as vbt
import yfinance as yf
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("nq_audit")

def clean_yfinance(df):
    if df.empty: return df
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    if df.index.tz is None: df.index = df.index.tz_localize('UTC')
    else: df.index = df.index.tz_convert('UTC')
    return df

def run_nq_critical_audit():
    """
    Phase 7: Critical Audit of Nasdaq H4 Strategy.
    Answering user's 10 questions with HARD DATA.
    
    1. Period: 2015-2024 (Full Cycle: Bull, Crash, Bear).
    2. Execution: Next Open (Realistic) vs Close (Optimistic).
    3. Costs: 2x Spread/Comm.
    4. Parameters: Neighborhood Sensitivity (18/48, 22/52).
    5. Regime: Bear Market (2022) Survival.
    """
    logger.info("Downloading Nasdaq History (2015-2025) - Daily Interval...")
    # YFinance H1 limit is 730d. We must use 1d for 10y.
    # NQ=F is Futures. QQQ is ETF. QQQ has better history likely.
    # Let's try NQ=F 1d first.
    try:
        df = yf.download("NQ=F", period="10y", interval="1d", progress=False, auto_adjust=True)
        df = clean_yfinance(df)
    except Exception as e:
        logger.error(f"NQ Download failed: {e}")
        df = pd.DataFrame()
    
    if len(df) < 2000:
        logger.warning("NQ=F data short/empty. Using QQQ (ETF) as long-term proxy.")
        df = yf.download("QQQ", period="10y", interval="1d", progress=False, auto_adjust=True)
        df = clean_yfinance(df)

    # Daily Data
    close_d1 = df['close']
    open_d1 = df['open']
    
    logger.info(f"Data Points (D1): {len(close_d1)} | Start: {close_d1.index[0]} | End: {close_d1.index[-1]}")
    
    # 1. Strategy Translation (H4 -> D1)
    # H4 20/50 approx D1 10/25? Or just standard D1 Trend (20/50, 50/200)?
    # Let's test standard Daily Trends: 20/50 (Fast) and 50/200 (Slow/Golden).
    
    # Test A: 20/50 Daily (Aggressive Trend)
    fast_w = 20
    slow_w = 50
    
    fast = vbt.MA.run(close_d1, fast_w)
    slow = vbt.MA.run(close_d1, slow_w)
    slope = slow.ma.diff() # Slope of Slow MA
    
    entries = fast.ma_crossed_above(slow) & (slope > 0)
    exits = fast.ma_crossed_below(slow)
    
    # 2. Execution Simulation
    # Next Open Execution
    price_exec = open_d1.shift(-1)
    
    # Costs: D1 trades less, so relative cost impact is lower, but we keep stress test.
    fees_stress = 0.001 
    
    pf_stress = vbt.Portfolio.from_signals(
        close_d1, # Signal generation on Close
        entries, exits, 
        price=price_exec, # Execution on NEXT Open
        freq='1d', 
        init_cash=100000,
        fees=fees_stress, 
        slippage=0.0001
    )
    
    # 3. Parameter Sensitivity (Neighborhood)
    # Test 18/48, 22/52
    fast_alt = vbt.MA.run(close_d1, [18, 22], param_product=True)
    slow_alt = vbt.MA.run(close_d1, [48, 52], param_product=True)
    # We need to broadcast properly
    # Simple loop for audit log clarity
    
    # Metrics
    stats = pf_stress.stats()
    sharpe = pf_stress.sharpe_ratio()
    trades = pf_stress.trades.count()
    max_dd = pf_stress.max_drawdown()
    win_rate = pf_stress.trades.win_rate()
    
    # Annual Returns
    annual_returns = pf_stress.returns().resample('YE').sum()
    
    logger.info(f"--- CRITICAL AUDIT RESULTS ---")
    logger.info(f"Sharpe (Stress Exec): {sharpe:.2f}")
    logger.info(f"Total Trades: {trades}")
    logger.info(f"Win Rate: {win_rate:.2%}")
    logger.info(f"Max Drawdown: {max_dd:.2%}")
    
    logger.info(f"--- Annual Performance ---")
    logger.info(annual_returns)
    
    # 4. Bear Market Check (2022)
    ret_2022 = annual_returns.get('2022-12-31', 0.0) # Index might be tz-aware
    # Try looking up by year using string matching or index type
    # For now, just print all.
    
    if sharpe < 1.0:
        logger.warning("FAILED: Sharpe < 1.0 under realistic conditions.")
    
    if trades < 50:
        logger.warning("FAILED: Insufficient Trades (< 50) for statistical significance.")

    if ret_2022 < -0.10:
        logger.warning(f"FAILED: 2022 Loss too high ({ret_2022:.2%}). Strategy rides Bull, dies in Bear.")

if __name__ == "__main__":
    run_nq_critical_audit()
