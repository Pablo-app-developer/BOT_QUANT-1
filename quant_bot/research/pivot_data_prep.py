
import yfinance as yf
import pandas as pd
import numpy as np
import logging
import os
from dateutil.relativedelta import relativedelta
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pivot_prep")

DATA_DIR = "quant_bot/data/pivot_ready"
os.makedirs(DATA_DIR, exist_ok=True)

TICKERS = ["BTC-USD", "ETH-USD", "NQ=F"] # Crypto + Tech

def prepare_pivot_datasets():
    """
    Day 1: Data Preparation.
    Specs:
    - 2 Years of 1h Data (730d).
    - Standardize: Open, High, Low, Close, Volume.
    - Timezone: UTC.
    - Splits: Train (60%), Val (20%), Test (20%).
    """
    
    for ticker in TICKERS:
        logger.info(f"Processing {ticker}...")
        
        # Download
        df = yf.download(ticker, period="730d", interval="1h", progress=False, auto_adjust=True)
        
        if df.empty:
            logger.error(f"Failed to download {ticker}")
            continue
            
        # Clean Columns
        # Standardize to lower case: open, high, low, close, volume
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        
        # Ensure UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
            
        df.index.name = 'datetime'
        
        # Select OHLCV
        cols = ['open', 'high', 'low', 'close', 'volume']
        # Check if volume exists
        if 'volume' not in df.columns:
            df['volume'] = 0
            
        df = df[cols]
        df = df.dropna()
        
        # Splits
        n = len(df)
        train_end = int(n * 0.60)
        val_end = int(n * 0.80)
        
        df['split'] = 'test'
        df.iloc[:train_end, df.columns.get_loc('split')] = 'train'
        df.iloc[train_end:val_end, df.columns.get_loc('split')] = 'val'
        
        stats = df['split'].value_counts(normalize=True)
        logger.info(f"Splits for {ticker}: \n{stats}")
        
        # Save
        filename = f"{DATA_DIR}/{ticker.replace('=','_').replace('-','_')}_H1.parquet"
        df.to_parquet(filename)
        logger.info(f"Saved: {filename}")
        
    logger.info("Day 1: Data Preparation Complete.")

if __name__ == "__main__":
    prepare_pivot_datasets()
