"""
main.py — Full pipeline orchestrator.

Runs end-to-end: data → features → HMM (full-sample) → walk-forward validation
→ portfolio optimization → backtest → performance summary and charts.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from hmmlearn import hmm

from data_pipeline      import load_data
from features           import compute_features
from regime_classifier  import fit_hmm, map_states_by_volatility, plot_regimes
from validation         import run_walk_forward
from backtest           import run_backtest, compute_metrics, plot_equity_curves


def print_section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def main():
    # ─── 1. Data ─────────────────────────────────────────────────────────────
    print_section("1 / 6   Data Ingestion")
    df = load_data(start_date="2015-01-01", end_date="2024-01-01")
    asset_returns = df[["SPY_ret", "TLT_ret", "GLD_ret"]].rename(
        columns={"SPY_ret": "SPY", "TLT_ret": "TLT", "GLD_ret": "GLD"}
    )
    print(f"Assets:  SPY | TLT | GLD | ^VIX")
    print(f"Period:  {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"Trading days: {len(df)}")

    # ─── 2. Features ─────────────────────────────────────────────────────────
    print_section("2 / 6   Feature Engineering")
    feat_df = compute_features(df)
    feature_cols = ["mom_21d_zscore", "vol_21d_zscore"]
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
    plot_regimes(feat_df, full_states, "regime_overlay.png")

    # ─── 4. Walk-Forward Validation ──────────────────────────────────────────
    print_section("4 / 6   Walk-Forward Validation")
    regimes = run_walk_forward(feat_df, feature_cols, n_splits=8,
                               min_train_size=252, test_size=63)
    print(f"\nOut-of-sample coverage: {len(regimes)} days")
    print("Out-of-sample regime distribution:")
    for k, v in regimes.map(label_map).value_counts().sort_index().items():
        print(f"  {k:<8}: {v} days")

    # ─── 5. Backtest ─────────────────────────────────────────────────────────
    print_section("5 / 6   Portfolio Optimization & Backtest")
    results = run_backtest(feat_df, regimes, asset_returns, tx_cost_bps=7)

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
    print("-" * 72)
    for label, ret_series in strategies.items():
        m = compute_metrics(ret_series, label=label)
        print(f"{m['Label']:<22} {m['Ann. Return']:>8} {m['Ann. Vol']:>7} "
              f"{m['Sharpe']:>7} {m['Sortino']:>8} {m['Max Drawdown']:>9} {m['Calmar']:>7}")

    # ─── Charts ───────────────────────────────────────────────────────────────
    plot_equity_curves(results, feat_df, regimes, "equity_curve.png")

    print("\nDone. Pipeline complete.")
    print("  Outputs saved:")
    print("    regime_overlay.png  - SPY price with Bull/Bear/Crisis shading")
    print("    equity_curve.png    - strategy vs benchmarks with drawdown panel")


if __name__ == "__main__":
    main()
