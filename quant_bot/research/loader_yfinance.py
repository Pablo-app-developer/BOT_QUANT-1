
import yfinance as yf
import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yfinance_loader")

DATA_DIR = "quant_bot/data/raw_yfinance"
os.makedirs(DATA_DIR, exist_ok=True)

def download_ticker(ticker: str, period: str = "2y", interval: str = "1h"):
    logger.info(f"Downloading {ticker} ({period}, {interval})...")
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    
    if df.empty:
        logger.error(f"No data found for {ticker}")
        return None
        
    logger.info(f"Downloaded {len(df)} rows.")
    # Standardize columns: Lowercase
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    
    # Save
    path = f"{DATA_DIR}/{ticker}_{interval}.parquet"
    df.to_parquet(path)
    logger.info(f"Saved to {path}")
    return df

if __name__ == "__main__":
    # Nasdaq 100 Futures, S&P 500 Futures, Bitcoin
    tickers = ["NQ=F", "ES=F", "BTC-USD"]
    
    # We try to get max intraday data allowed by YF
    # 60 days max for 15m/5m?
    # 730 days max for 1h.
    
    for t in tickers:
        download_ticker(t, period="730d", interval="1h")
