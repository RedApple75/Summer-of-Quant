"""
main.py — Full pipeline orchestrator.

Runs end-to-end: data (2005-2024) → features → HMM (full-sample) → walk-forward validation (probabilities)
→ portfolio optimization (blended) → backtest (smoothed) → performance summary and charts.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from data_pipeline      import load_data
from features           import compute_features
from regime_classifier  import fit_hmm, map_states_by_volatility, plot_regimes
from validation         import run_walk_forward_probabilities
from backtest           import run_backtest_probabilities, compute_metrics, plot_equity_curves


def print_section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def main():
    # Ensure charts directory exists
    os.makedirs("charts", exist_ok=True)

    # ─── 1. Data ─────────────────────────────────────────────────────────────
    print_section("1 / 6   Data Ingestion")
    df = load_data(start_date="2005-01-01", end_date="2024-01-01")
    asset_returns = df[["SPY_ret", "TLT_ret", "GLD_ret", "BIL_ret"]].rename(
        columns={"SPY_ret": "SPY", "TLT_ret": "TLT", "GLD_ret": "GLD", "BIL_ret": "BIL"}
    )
    print(f"Assets:  SPY | TLT | GLD | BIL | ^VIX | ^TNX")
    print(f"Period:  {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"Trading days: {len(df)}")

    # ─── 2. Features ─────────────────────────────────────────────────────────
    print_section("2 / 6   Feature Engineering")
    feat_df = compute_features(df)
    feature_cols = ["mom_21d_zscore", "VIX_zscore"]
    
    # Align returns
    common_idx = feat_df.index.intersection(asset_returns.index)
    feat_df = feat_df.loc[common_idx]
    asset_returns = asset_returns.loc[common_idx]
    
    print(f"Feature rows after NaN drop: {len(feat_df)}")
    print(f"Features fed to HMM: {feature_cols}")

    # ─── 3. Full-Sample HMM (sanity check / visualization only) ──────────────
    print_section("3 / 6   HMM Regime Classifier (full-sample, visualization only)")
    model = fit_hmm(feat_df, feature_cols)
    full_states, state_map = map_states_by_volatility(model, feat_df, feature_cols)
    label_map = {0: "Bull", 1: "Bear", 2: "Crisis"}
    print("Transition Probability Matrix:")
    print(np.round(model.transmat_, 4))
    print("\nFull-sample regime distribution:")
    for k, v in full_states.map(label_map).value_counts().sort_index().items():
        print(f"  {k:<8}: {v} days")
    
    # Save regime overlay to charts directory
    plot_regimes(feat_df, full_states, "charts/regime_overlay.png")

    # ─── 4. Walk-Forward Validation ──────────────────────────────────────────
    print_section("4 / 6   Walk-Forward Validation (Probabilities)")
    regime_probs = run_walk_forward_probabilities(
        feat_df, feature_cols, n_splits=12, min_train_size=504, test_size=126
    )
    print(f"\nOut-of-sample coverage: {len(regime_probs)} days")
    print("Regime probability stats:")
    print(regime_probs.describe())

    # ─── 5. Backtest ─────────────────────────────────────────────────────────
    print_section("5 / 6   Portfolio Optimization & Backtest (Blended)")
    # Extract TNX series for ZLB filter
    tnx_series = df["TNX"].loc[regime_probs.index]
    
    # Using HMM H settings: 1.5x Bull leverage, 20% short SPY, ZLB active, asymmetric smoothing, 0.5% hurdle
    results = run_backtest_probabilities(
        feat_df, regime_probs, asset_returns, lookback=252, tx_cost_bps=7,
        alpha_smooth=0.03, cash_overlay=True, tnx_series=tnx_series, zlb_active=True,
        leverage_factor=1.5, short_equity_allocation=0.20, borrow_cost_ann=0.04,
        asymmetric_smoothing=True
    )

    # ─── 6. Performance Summary ───────────────────────────────────────────────
    print_section("6 / 6   Performance Summary")
    strategies = {
        "Dynamic (gross)":    results["strategy_gross"],
        "Dynamic (7bps net)": results["strategy_net"],
        "Static 60/40":       results["static_6040"],
        "Equal Weight":       results["equal_weight"],
    }

    print(f"\n{'Strategy':<22} {'Ann.Ret':>8} {'Vol':>7} {'Sharpe':>7} "
          f"{'Sortino':>8} {'MaxDD':>9} {'Calmar':>7}")
    # Fix formatting print using compute_metrics labels
    print("-" * 72)
    for label, ret_series in strategies.items():
        m = compute_metrics(ret_series, label=label)
        print(f"{m['Label']:<22} {m['Ann. Return']:>8} {m['Ann. Vol']:>7} "
              f"{m['Sharpe']:>7} {m['Sortino']:>8} {m['Max Drawdown']:>9} {m['Calmar']:>7}")

    # ─── Charts ───────────────────────────────────────────────────────────────
    plot_equity_curves(results, feat_df, regime_probs, "charts/equity_curve.png")

    print("\nDone. Pipeline complete.")
    print("  Outputs saved:")
    print("    charts/regime_overlay.png  - SPY price with Bull/Bear/Crisis shading")
    print("    charts/equity_curve.png    - strategy vs benchmarks with drawdown panel")


if __name__ == "__main__":
    main()
