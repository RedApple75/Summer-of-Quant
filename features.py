import pandas as pd
import numpy as np

def compute_features(df, min_periods=63):
    """
    Computes momentum, volatility, and safe expanding z-score features.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame from data_pipeline containing asset prices and returns.
        min_periods (int): Minimum lookback period for expanding window normalization.
        
    Returns:
        pd.DataFrame: DataFrame with engineered and normalized features.
    """
    print("Computing features...")
    features_df = df.copy()
    
    # We will compute features for SPY as our primary regime signal asset
    # (since the engine's regime reflects the overall equity market mood)
    
    # 1. Momentum: simple return over N days
    for window in [5, 21, 63, 126]:
        features_df[f"mom_{window}d"] = features_df["SPY"].pct_change(window)
        
    # 2. Volatility: rolling std dev of log returns, annualized
    for window in [5, 21, 63]:
        features_df[f"vol_{window}d"] = features_df["SPY_ret"].rolling(window).std() * np.sqrt(252)
        
    # 3. Z-Scoring (Safe/Expanding window to prevent lookahead bias)
    # List of raw features to scale
    feature_cols = [
        "mom_5d", "mom_21d", "mom_63d", "mom_126d",
        "vol_5d", "vol_21d", "vol_63d", "VIX"
    ]
    
    # We drop initial NaNs from rolling/momentum calculations before z-scoring
    # Max rolling window is 126, so the first 126 rows will contain NaNs
    features_df = features_df.dropna(subset=feature_cols)
    
    # Apply expanding z-scoring
    for col in feature_cols:
        expanding_mean = features_df[col].expanding(min_periods=min_periods).mean()
        expanding_std = features_df[col].expanding(min_periods=min_periods).std()
        
        # Calculate z-score using only historical data up to time t
        features_df[f"{col}_zscore"] = (features_df[col] - expanding_mean) / expanding_std
        
    # Drop rows that don't have enough history for the expanding z-score
    # (which will be the first min_periods - 1 rows of the features_df)
    zscore_cols = [f"{col}_zscore" for col in feature_cols]
    features_df = features_df.dropna(subset=zscore_cols)
    
    print(f"Features computed. Shape of features dataset: {features_df.shape}")
    return features_df

if __name__ == "__main__":
    from data_pipeline import load_data
    df = load_data()
    feat_df = compute_features(df)
    
    print("\nHead of computed features:")
    cols_to_print = ["mom_21d", "mom_21d_zscore", "vol_21d", "vol_21d_zscore"]
    print(feat_df[cols_to_print].head(10))
    print("\nTail of computed features:")
    print(feat_df[cols_to_print].tail(10))
