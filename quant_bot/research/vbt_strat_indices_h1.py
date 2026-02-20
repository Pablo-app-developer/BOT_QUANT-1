
import vectorbt as vbt
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vbt_indices")

def load_data(ticker="NQ=F"):
    path = f"quant_bot/data/raw_yfinance/{ticker}_1h.parquet"
    try:
        df = pd.read_parquet(path)
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Check if columns are MultiIndex (common with yfinance download)
        if isinstance(df.columns, pd.MultiIndex):
             # Collapse or select correct level. Usually Level 1 is Ticker
             # But my loader lowercased columns?
             # Let's check if 'close' exists
             pass
        
        return df['close']
    except Exception as e:
        logger.error(f"Failed to load {ticker}: {e}")
        return None

def run_indices_trend():
    tickers = ["NQ=F", "ES=F", "BTC-USD"]
    
    for t in tickers:
        logger.info(f"--- Strategy Test: {t} (1H Data) ---")
        close = load_data(t)
        if close is None: continue
        
        # Strategy: Simple MA Crossover (Trend Following)
        # 1. Fast MA (10, 20, 50)
        # 2. Slow MA (50, 100, 200)
        
        fast_windows = [10, 20, 50]
        slow_windows = [50, 100, 200]
        
        # Generate Signals
        # We iterate manually to keep it simple and clean
        best_sharpe = -999
        best_params = None
        
        for f in fast_windows:
            for s in slow_windows:
                if f >= s: continue
                
                fast = vbt.MA.run(close, f)
                slow = vbt.MA.run(close, s)
                
                entries = fast.ma_crossed_above(slow)
                exits = fast.ma_crossed_below(slow)
                
                pf = vbt.Portfolio.from_signals(
                    close,
                    entries=entries,
                    exits=exits,
                    short_entries=exits,
                    short_exits=entries,
                    freq='1h',
                    init_cash=100000,
                    # Fees for futures/crypto ~ 0.05%?
                    fees=0.0005, 
                    slippage=0.0001
                )
                
                sharpe = pf.sharpe_ratio()
                ret = pf.total_return()
                dd = pf.max_drawdown()
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = (f, s)
                    best_ret = ret
                    best_dd = dd
        
        logger.info(f"[{t}] Best Result (MA {best_params}): Sharpe={best_sharpe:.2f} | Return={best_ret:.2%} | DD={best_dd:.2%}")

if __name__ == "__main__":
    run_indices_trend()
