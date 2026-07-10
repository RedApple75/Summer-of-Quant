import numpy as np
import pandas as pd
from hmmlearn import hmm


def expanding_walk_forward_splits(n_obs, n_splits=5, min_train_size=252, test_size=63):
    """
    Yields (train_indices, test_indices) for an expanding-window walk-forward split.

    At each fold the training set starts at 0 and expands by test_size. Test sets are
    strictly non-overlapping and always come after the training set — no future leakage.
    """
    step = (n_obs - min_train_size) // n_splits
    for i in range(n_splits):
        train_end = min_train_size + i * step
        test_end  = min(train_end + test_size, n_obs)
        if train_end >= n_obs:
            break
        train_idx = np.arange(0, train_end)
        test_idx  = np.arange(train_end, test_end)
        yield train_idx, test_idx


def run_walk_forward(features_df, feature_cols, n_splits=5, min_train_size=252,
                     test_size=63, n_components=3, random_state=42):
    """
    Runs the walk-forward validation loop.

    For each fold:
      1. Scales features using ONLY training-set mean/std (preventing lookahead bias).
      2. Fits the HMM on scaled training data.
      3. Predicts regimes on the test slice using the fitted model.

    Returns a DataFrame aligned with features_df containing out-of-sample regime labels
    and the volatility-mapped state assignment for every test day.
    """
    n_obs = len(features_df)
    splits = list(expanding_walk_forward_splits(n_obs, n_splits, min_train_size, test_size))

    all_test_states  = pd.Series(index=features_df.index, dtype=float)
    all_test_regimes = pd.Series(index=features_df.index, dtype=float)

    print(f"Running {len(splits)}-fold walk-forward validation...")

    for fold, (train_idx, test_idx) in enumerate(splits):
        train_df = features_df.iloc[train_idx]
        test_df  = features_df.iloc[test_idx]

        # --- Scale: fit mu/sigma on train only, apply to both ---
        mu    = train_df[feature_cols].mean()
        sigma = train_df[feature_cols].std().replace(0, 1)   # guard against zero-std

        X_train = ((train_df[feature_cols] - mu) / sigma).values
        X_test  = ((test_df[feature_cols]  - mu) / sigma).values

        # Verify no leakage: test z-scores computed with train stats differ from
        # what full-dataset stats would give — print this diff as a sanity check.
        full_mu    = features_df[feature_cols].mean()
        full_sigma = features_df[feature_cols].std().replace(0, 1)
        X_test_leaky = ((test_df[feature_cols] - full_mu) / full_sigma).values
        max_diff = np.abs(X_test - X_test_leaky).max()

        # --- Fit HMM on training data only ---
        model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            n_iter=100,
            random_state=random_state
        )
        model.fit(X_train)

        # --- Predict on test ---
        raw_states = model.predict(X_test)

        # Map states: lowest avg vol -> Bull(0), middle -> Bear(1), highest -> Crisis(2)
        temp = pd.DataFrame({"state": raw_states, "vol": test_df["vol_21d"].values})
        state_vol_mean  = temp.groupby("state")["vol"].mean().sort_values()
        state_map = {orig: mapped for mapped, orig in enumerate(state_vol_mean.index)}
        mapped = pd.Series(raw_states, index=test_df.index).map(state_map)

        all_test_states.iloc[test_idx]  = raw_states
        all_test_regimes.iloc[test_idx] = mapped.values

        test_start = features_df.index[test_idx[0]].date()
        test_end   = features_df.index[test_idx[-1]].date()
        regime_counts = mapped.value_counts().sort_index().to_dict()
        label_map = {0: "Bull", 1: "Bear", 2: "Crisis"}
        regime_str = ", ".join(f"{label_map.get(k,'?')}:{v}" for k, v in sorted(regime_counts.items()))
        print(f"  Fold {fold+1}: train[0..{len(train_idx)}] | "
              f"test[{test_start}->{test_end}] | "
              f"leakage_diff={max_diff:.4f} | {regime_str}")

    return all_test_regimes.dropna()


def run_walk_forward_probabilities(features_df, feature_cols, n_splits=5, min_train_size=252,
                                  test_size=63, n_components=3, random_state=42):
    """
    Runs walk-forward validation and returns posterior probabilities of each state
    mapped consistently to Bull (col 0), Bear (col 1), and Crisis (col 2).
    """
    n_obs = len(features_df)
    splits = list(expanding_walk_forward_splits(n_obs, n_splits, min_train_size, test_size))

    # Output DataFrame for probabilities
    prob_cols = ["Bull_prob", "Bear_prob", "Crisis_prob"]
    all_probs = pd.DataFrame(index=features_df.index, columns=prob_cols, dtype=float)

    for fold, (train_idx, test_idx) in enumerate(splits):
        train_df = features_df.iloc[train_idx]
        test_df  = features_df.iloc[test_idx]

        mu    = train_df[feature_cols].mean()
        sigma = train_df[feature_cols].std().replace(0, 1)

        X_train = ((train_df[feature_cols] - mu) / sigma).values
        X_test  = ((test_df[feature_cols]  - mu) / sigma).values

        model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            n_iter=100,
            random_state=random_state
        )
        model.fit(X_train)

        # Get state probabilities
        raw_probs = model.predict_proba(X_test)  # Shape (N_test, n_components)
        raw_states = model.predict(X_test)

        # Build mapping based on training average volatility
        # Map each raw state to Bull/Bear/Crisis based on volatility of that state on the training set
        train_state_vol = []
        train_raw_states = model.predict(X_train)
        temp_train = pd.DataFrame({"state": train_raw_states, "vol": train_df["vol_21d"].values})
        state_vol_mean = temp_train.groupby("state")["vol"].mean()
        
        # If any component wasn't predicted in training, give it a default ordering
        for comp in range(n_components):
            if comp not in state_vol_mean:
                state_vol_mean[comp] = 999.0 if comp == 2 else (0.0 if comp == 0 else 0.1)
                
        state_vol_mean = state_vol_mean.sort_values()
        # map: raw state -> sorted index (0=lowest vol, 1=med, 2=highest)
        state_map = {orig: mapped for mapped, orig in enumerate(state_vol_mean.index)}

        # Reorder probabilities according to state_map
        mapped_probs = np.zeros_like(raw_probs)
        for orig, mapped in state_map.items():
            mapped_probs[:, mapped] = raw_probs[:, orig]

        all_probs.iloc[test_idx] = mapped_probs

    return all_probs.dropna()



if __name__ == "__main__":
    from data_pipeline import load_data
    from features import compute_features

    df      = load_data()
    feat_df = compute_features(df)

    feature_cols = ["mom_21d_zscore", "vol_21d_zscore"]
    regimes = run_walk_forward(feat_df, feature_cols)

    print(f"\nOut-of-sample regime labels: {len(regimes)} days covered")
    print("Regime distribution:")
    label_map = {0: "Bull", 1: "Bear", 2: "Crisis"}
    print(regimes.map(label_map).value_counts())
