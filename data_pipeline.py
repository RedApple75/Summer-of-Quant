import yfinance as yf
import pandas as pd
import numpy as np
import os

def load_data(start_date="2015-01-01", end_date="2024-01-01"):
    """
    Downloads, cleans, and aligns daily asset prices (SPY, TLT, GLD, BIL) and VIX/TNX.
    """
    print(f"Downloading historical data from {start_date} to {end_date}...")
    tickers = ["SPY", "TLT", "GLD", "BIL", "^VIX", "^TNX"]
    
    # Download raw adjusted close prices
    raw = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
    
    # Extract Close prices
    if "Close" in raw.columns.levels[0]:
        close_prices = raw["Close"].copy()
    else:
        close_prices = raw.copy()
        
    close_prices.columns.name = None
    close_prices = close_prices.rename(columns={"^VIX": "VIX", "^TNX": "TNX"})
    close_prices = close_prices.dropna(how="all")
    
    # Handle pre-inception data for BIL (launched May 2007) and yields
    close_prices["BIL"] = close_prices["BIL"].bfill()
    close_prices["TNX"] = close_prices["TNX"].ffill().bfill()
    close_prices["VIX"] = close_prices["VIX"].ffill()
    
    # Compute log returns for assets
    for asset in ["SPY", "TLT", "GLD", "BIL"]:
        close_prices[f"{asset}_ret"] = np.log(close_prices[asset]).diff()
        
    # Drop initial row because log returns will have NaN at start
    final_df = close_prices.dropna(subset=["SPY", "TLT", "GLD", "VIX", "TNX", "SPY_ret", "TLT_ret", "GLD_ret"])
    
    print(f"Data downloaded successfully. Total observations: {len(final_df)}")
    return final_df

if __name__ == "__main__":
    df = load_data()
    print("\nHead of aligned dataset:")
    print(df.head())
    print("\nMissing values check:")
    print(df.isna().sum())
