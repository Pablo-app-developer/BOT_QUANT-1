
import vectorbt as vbt
import pandas as pd
import numpy as np
import logging
from data import loader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("vbt_h4_mom")

def load_clean_data():
    df = loader.load_processed()
    df = df[df.index.year < 2025] # Exclude 2025
    return df['close']

def run_h4_momentum():
    close_m1 = load_clean_data()
    
    # Scale Up: H4 and D1
    timeframes = ['4H', '1D']
    
    # Donchian Parameters (Lookback)
    windows = [20, 50, 100, 200]
    
    for tf in timeframes:
        logger.info(f"--- Testing Timeframe: {tf} ---")
        close_tf = close_m1.resample(tf).last().dropna()
        
        # Donchian Channels (Max/Min of last N bars)
        # VBT doesn't have explicit Donchian, but we can use rolling max/min
        # rolling_max(window).shift(1) to avoid lookahead for breakout
        
        for w in windows:
            # Donchian Upper: Max of previous W bars
            upper = close_tf.rolling(w).max().shift(1)
            lower = close_tf.rolling(w).min().shift(1)
            
            # Long: Close > Upper (Breakout)
            entries = close_tf > upper
            
            # Short: Close < Lower (Breakdown)
            entries_short = close_tf < lower
            
            # Exits:
            # Option A: Reversal (Always In) -> entries_short is exit for long
            # Option B: Trailing Stop?
            # Let's try "Always In" first (Donchian Trend Following)
            
            # We need to clean signals (remove consecutive entries if already in)
            # VBT Portfolio handles clean_signals=True by default?
            
            pf = vbt.Portfolio.from_signals(
                close_tf,
                entries=entries,
                exits=entries_short,
                short_entries=entries_short,
                short_exits=entries,
                freq=tf,
                init_cash=10000,
                # Costs: Spread is less relevant in H4/D1 but still pays.
                # 1.2 pips
                fees=0.00012,
                slippage=0.00005
            )
            
            total_return = pf.total_return()
            sharpe = pf.sharpe_ratio()
            trades = pf.trades.count()
            max_dd = pf.max_drawdown()
            
            if sharpe > 0.5:
                # We want robust trends.
                logger.info(f"TF: {tf} | Donchian: {w} | Sharpe: {sharpe:.2f} | Return: {total_return:.2%} | Trades: {trades} | MaxDD: {max_dd:.2%}")

if __name__ == "__main__":
    run_h4_momentum()
