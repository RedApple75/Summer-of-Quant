import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from hmmlearn import hmm
import os

def fit_hmm(features_df, feature_cols, n_components=3, random_state=42):
    """
    Fits a Gaussian Hidden Markov Model (HMM) on standard features.
    """
    print(f"Fitting GaussianHMM with {n_components} components...")
    X = features_df[feature_cols].values
    
    # We use diagonal covariance (diag) to avoid overfitting, with a fixed random state
    model = hmm.GaussianHMM(
        n_components=n_components, 
        covariance_type="diag", 
        n_iter=100, 
        random_state=random_state
    )
    model.fit(X)
    return model

def map_states_by_volatility(model, features_df, feature_cols):
    """
    Predicts hidden states and maps them dynamically such that:
      - State 0: Bull (lowest volatility)
      - State 1: Bear (medium volatility)
      - State 2: Crisis (highest volatility)
      
    Returns:
      pd.Series: The mapped state sequence with index aligned to features_df.
      dict: The state map (original index -> mapped index).
    """
    X = features_df[feature_cols].values
    predicted_states = model.predict(X)
    
    # Create temp dataframe to calculate mean volatility for each state
    temp = pd.DataFrame({
        "state": predicted_states,
        "vol": features_df["vol_21d"]  # Use raw volatility for intuitive mapping
    })
    
    # Sort states by mean volatility
    state_vol_means = temp.groupby("state")["vol"].mean().sort_values()
    
    # Map: lowest vol -> 0, middle vol -> 1, highest vol -> 2
    state_map = {orig_state: mapped_state for mapped_state, orig_state in enumerate(state_vol_means.index)}
    
    mapped_states = pd.Series(predicted_states, index=features_df.index).map(state_map)
    return mapped_states, state_map

def plot_regimes(df, state_series, output_path="regime_overlay.png"):
    """
    Plots the price chart with background colors indicating the detected regime.
    """
    print(f"Generating regime overlay plot and saving to {output_path}...")
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot SPY price
    ax.plot(df.index, df["SPY"], color="black", lw=1.5, label="SPY Price")
    
    # Overlay regimes
    # Colors: Bull (light green), Bear (light yellow), Crisis (light red)
    colors = {0: "#a1d99b", 1: "#fdae61", 2: "#d73027"}
    labels = {0: "Bull (Low Vol)", 1: "Bear (Med Vol)", 2: "Crisis (High Vol)"}
    
    # Track which labels have been added to the legend
    legend_added = set()
    
    for i in range(len(df) - 1):
        state = state_series.iloc[i]
        start_date = df.index[i]
        end_date = df.index[i+1]
        
        # Color span
        label = labels[state] if state not in legend_added else None
        ax.axvspan(start_date, end_date, color=colors[state], alpha=0.3, label=label)
        legend_added.add(state)
        
    ax.set_title("SPY Price with Inferred Market Regimes (HMM)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("SPY Adjusted Close ($)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    from data_pipeline import load_data
    from features import compute_features
    
    df = load_data()
    feat_df = compute_features(df)
    
    # Use mom_21d_zscore and vol_21d_zscore as features for training
    feature_cols = ["mom_21d_zscore", "vol_21d_zscore"]
    
    model = fit_hmm(feat_df, feature_cols)
    states, state_map = map_states_by_volatility(model, feat_df, feature_cols)
    
    # Show transition matrix
    print("\nTransition Probability Matrix:")
    print(model.transmat_)
    
    # Show count of days in each regime
    print("\nRegime Distribution (Days):")
    print(states.value_counts().sort_index())
    
    # Generate verification plot
    plot_regimes(feat_df, states, "regime_overlay.png")
