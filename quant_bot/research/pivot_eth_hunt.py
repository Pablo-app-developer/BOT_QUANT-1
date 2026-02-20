
import vectorbt as vbt
import pandas as pd
import numpy as np
import logging
from research.pivot_baselines import load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("eth_hunt")

def run_eth_deep_dive():
    """
    Day 3: ETH "Rescue Mission".
    Baseline ETH Trend (MA 50/100) failed.
    We need to find if ANY strategy works on ETH H1 (Train Set).
    
    Hypotheses to Test:
    1.  **Faster Trend**: EMA 10/20, 20/50.
    2.  **Trend + Filter**: MA Cross + ADX > 25 (Strong Trend).
    3.  **Volatility Breakout (Keltner)**: Price > EMA + 2*ATR.
    4.  **Momentum**: RSI > 50 (Bullish) & RSI > RSI[1] (Rising).
    """
    logger.info("Starting ETH Deep Dive on TRAIN set...")
    
    # Load ETH Data
    df = load_data("ETH_USD")
    train_df = df[df['split'] == 'train']
    close = train_df['close']
    high = train_df['high']
    low = train_df['low']
    
    logger.info(f"ETH Train Samples: {len(close)}")
    
    results = []
    
    # 1. EMA Trend (Faster)
    fast_windows = [5, 10, 20]
    slow_windows = [20, 50, 100]
    
    for f in fast_windows:
        for s in slow_windows:
            if f >= s: continue
            
            fast = vbt.MA.run(close, f, ewm=True) # EMA
            slow = vbt.MA.run(close, s, ewm=True) # EMA
            
            entries = fast.ma_crossed_above(slow)
            exits = fast.ma_crossed_below(slow)
            
            pf = vbt.Portfolio.from_signals(close, entries, exits, freq='1h', fees=0.0006, slippage=0.0001)
            results.append({'Strat': f'EMA({f},{s})', 'Sharpe': pf.sharpe_ratio()})

    # 2. Trend + ADX Filter
    # VBT doesn't have built-in ADX easily accessible in 1 line?
    # We can use pandas_ta or calculate manual.
    # ADX: Needs TR, DM+, DM-.
    # Let's try a simplified filter: Volatility Filter (ATR).
    # Only trade if ATR(14) > Threshold?
    # Or simpler: RSI Trend confirmation.
    
    # 3. Keltner Channel Breakout
    # EMA(20) +/- 2*ATR(10)
    # Using vbt.BBANDS as proxy? No, BB uses StdDev. 
    # Let's construct manually.
    
    ma = vbt.MA.run(close, 20, ewm=True).ma
    atr = vbt.ATR.run(high, low, close, 10).atr
    
    k_upper = ma + 2 * atr
    k_lower = ma - 2 * atr
    
    entries_k = close > k_upper
    exits_k = close < k_lower
    pf_k = vbt.Portfolio.from_signals(close, entries_k, exits_k, freq='1h', fees=0.0006, slippage=0.0001)
    results.append({'Strat': 'Keltner(20,2.0)', 'Sharpe': pf_k.sharpe_ratio()})
    
    # 4. RSI Momentum (50 Cross)
    # Long if RSI crosses above 50
    # Short if RSI crosses below 50
    rsi = vbt.RSI.run(close, 14)
    entries_r = rsi.rsi_crossed_above(50)
    exits_r = rsi.rsi_crossed_below(50)
    pf_r = vbt.Portfolio.from_signals(close, entries_r, exits_r, freq='1h', fees=0.0006, slippage=0.0001)
    results.append({'Strat': 'RSI(50 Cross)', 'Sharpe': pf_r.sharpe_ratio()})

    # Report Top 5
    res_df = pd.DataFrame(results).sort_values('Sharpe', ascending=False)
    logger.info(f"Top ETH Strategies:\n{res_df.head(5)}")
    
    best = res_df.iloc[0]
    if best['Sharpe'] > 0.8:
        logger.info(f"SUCCESS: Found candidate {best['Strat']} (Sharpe {best['Sharpe']:.2f})")
    else:
        logger.info(f"FAILURE: Best Sharpe {best['Sharpe']:.2f} is weak.")

if __name__ == "__main__":
    run_eth_deep_dive()
