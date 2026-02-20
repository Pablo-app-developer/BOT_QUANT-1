
import vectorbt as vbt
import pandas as pd
import numpy as np
import logging
from data import loader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("vbt_breakout")

def load_clean_data():
    df = loader.load_processed()
    df = df[df.index.year < 2025] # Exclude 2025
    return df['close']

def run_breakout_strategy():
    close_m1 = load_clean_data()
    
    # Timeframes to Test
    timeframes = ['15T', '30T', '1H']
    
    # Parameters to Optimize
    windows = [20, 50, 100]
    stds = [2.0, 2.5, 3.0, 4.0]
    
    # Iterate Timeframes
    for tf in timeframes:
        logger.info(f"--- Testing Timeframe: {tf} ---")
        close_tf = close_m1.resample(tf).last().dropna()
        
        # Run BB Indicator for all windows/stds
        # VBT supports broadcasting. We can pass lists.
        # fast_ma = vbt.MA.run(close_tf, window=windows) -> Generates columns for each window
        
        # BB Run
        # We use param_product=True to get all combinations of window/std
        bb = vbt.BBANDS.run(close_tf, window=windows, alpha=stds, param_product=True)
        
        # Signal: Breakout Upper (Long) or Lower (Short)
        # Signal: Breakout Upper (Long) or Lower (Short)
        # Long: Close > Upper
        entries = close_tf.vbt > bb.upper
        # Short: Close < Lower
        exits = close_tf.vbt < bb.lower
        
        # Setup Portfolio
        # We reverse position on opposite signal? Or Fixed SL/TP?
        # Let's try "Reversal" first (Always in market if signal exists)
        # entries and exits here are strictly breakout signals.
        # If we want pure reversal:
        short_entries = exits
        long_entries = entries
        
        # Clean signals (remove consecutive)
        # vbt.Portfolio.from_signals handles this?
        
        pf = vbt.Portfolio.from_signals(
            close_tf,
            entries=long_entries,
            short_entries=short_entries,
            # Exit on opposite signal
            exits=None, 
            short_exits=None,
            
            # Costs
            fees=0.00012, # 1.2 pips
            slippage=0.00005, # 0.5 pips
            freq=tf,
            init_cash=10000
        )
        
        # Analysis
        total_return = pf.total_return()
        sharpe = pf.sharpe_ratio()
        max_dd = pf.max_drawdown()
        trades = pf.trades.count()
        
        # Combine metrics into DataFrame
        metrics = pd.DataFrame({
            'Total Return': total_return,
            'Sharpe': sharpe,
            'Max DD': max_dd,
            'Trades': trades
        })
        
        # Filter for "Decent" results
        # Sharpe > 0.5?
        best = metrics[metrics['Sharpe'] > 0.5]
        
        if not best.empty:
            logger.info(f"Found {len(best)} candidates in {tf}!")
            print(best.sort_values('Sharpe', ascending=False).head(5))
        else:
            logger.info(f"No candidates found in {tf} (Best Sharpe: {metrics['Sharpe'].max():.2f})")

if __name__ == "__main__":
    run_breakout_strategy()
