
import vectorbt as vbt
import yfinance as yf
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vbt_btc_m5")

def run_btc_m5():
    logger.info("Downloading BTC-USD (M5, Last 60 Days)...")
    df = yf.download("BTC-USD", period="60d", interval="5m", progress=False, auto_adjust=True)
    
    if df.empty:
        logger.error("Download failed.")
        return
        
    # Columns check
    # MultiIndex or Flat?
    # Usually: (Price, Ticker) MultiIndex?
    # Let's inspect columns
    # My simple logic: Just grab 'Close'
    try:
        if isinstance(df.columns, pd.MultiIndex):
            # Select 'Close' level explicitly if possible
            if 'Close' in df.columns.get_level_values(0):
                close = df.xs('Close', axis=1, level=0)
            else:
                 # Try to just take column 3? (Open, High, Low, Close)
                 # Close is usually 3?
                 logger.warning(f"Columns: {df.columns}")
                 # Maybe lower?
                 pass
            # Usually yfinance returns (Price, Ticker). So df['Close'] gives DataFrame with Tickers as columns.
            # If only one ticker, df['Close'] is series? Or DataFrame with 1 col?
            close = df['Close']
            if isinstance(close, pd.DataFrame):
                 close = close.iloc[:, 0] # Take first column (BTC-USD)
        else:
            close = df['Close']
            
    except Exception as e:
        logger.error(f"Column error: {e}")
        # Fallback
        close = df.iloc[:, 3] # Guess Close
        
    logger.info(f"Loaded {len(close)} M5 bars.")
    
    # Strategy: Intraday Trend
    # Fast MA: 10, 20, 50
    # Slow MA: 50, 100, 200
    
    fast_windows = [10, 20, 50]
    slow_windows = [50, 100, 200]
    
    best_sharpe = -999
    
    for f in fast_windows:
        for s in slow_windows:
            if f >= s: continue
            
            fast = vbt.MA.run(close, f)
            slow = vbt.MA.run(close, s)
            
            entries = fast.ma_crossed_above(slow)
            exits = fast.ma_crossed_below(slow)
            
            pf = vbt.Portfolio.from_signals(
                close, entries, exits, 
                short_entries=exits, short_exits=entries,
                freq='5min', init_cash=10000,
                fees=0.0005, slippage=0.0001
            )
            
            if pf.sharpe_ratio() > best_sharpe:
                best_sharpe = pf.sharpe_ratio()
                best_ret = pf.total_return()
                best_dd = pf.max_drawdown()
                best_params = (f, s)
    
    logger.info(f"BTC M5 Trend Best Result: MA {best_params} | Sharpe: {best_sharpe:.2f} | Return: {best_ret:.2%} | MaxDD: {best_dd:.2%}")
    
    if best_sharpe > 1.0:
        logger.info("SUCCESS! Found Intraday Edge on BTC M5.")
    else:
        logger.info("FAILURE. No Edge even on BTC M5.")

if __name__ == "__main__":
    run_btc_m5()
