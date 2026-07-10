import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ─── Portfolio Optimizers ────────────────────────────────────────────────────

def max_sharpe_weights(mu, cov, rf=0.0):
    """
    Maximize Sharpe ratio via the standard convex reformulation.
    Scales so that (mu - rf)^T y = 1, then minimizes y^T Σ y, then normalizes.
    Falls back to equal weights if the problem is infeasible.
    """
    n   = len(mu)
    y   = cp.Variable(n)
    excess = (mu - rf).values

    objective   = cp.Minimize(cp.quad_form(y, cov.values))
    constraints = [excess @ y == 1, y >= 0]
    prob        = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, warm_start=True)

    if prob.status not in ("optimal", "optimal_inaccurate") or y.value is None:
        return np.ones(n) / n          # equal weight fallback

    raw = y.value
    raw = np.maximum(raw, 0)
    total = raw.sum()
    if total <= 1e-8:
        return np.ones(n) / n
    return raw / total


def min_variance_weights(cov):
    """
    Minimize portfolio variance with no-short-selling and full-investment constraints.
    """
    n   = len(cov)
    w   = cp.Variable(n)
    objective   = cp.Minimize(cp.quad_form(w, cov.values))
    constraints = [cp.sum(w) == 1, w >= 0]
    prob        = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, warm_start=True)

    if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        return np.ones(n) / n
    return np.maximum(w.value, 0) / np.maximum(w.value, 0).sum()


def risk_parity_weights(cov, tol=1e-6, max_iter=500):
    """
    Equal Risk Contribution (risk parity) via iterative (Spinu) algorithm.
    Target: each asset contributes equally to total portfolio variance.
    Falls back to equal weights if it does not converge.
    """
    n    = len(cov)
    cov_arr = cov.values
    w    = np.ones(n) / n

    for _ in range(max_iter):
        port_var   = w @ cov_arr @ w
        marginal   = cov_arr @ w
        risk_contrib = w * marginal / port_var
        # Move weights toward equal risk contribution
        w_new = w * (1.0 / (n * risk_contrib + 1e-12))
        w_new /= w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            return w_new
        w = w_new

    return w   # return best estimate even if not converged


def get_regime_weights(regime, mu, cov):
    """
    Route to the appropriate optimizer for each regime.
    Bull  → Maximize Sharpe ratio   (take equity risk)
    Bear  → Risk parity             (spread risk equally)
    Crisis→ Minimize variance       (capital preservation)
    """
    if regime == 0:    # Bull
        return max_sharpe_weights(mu, cov)
    elif regime == 1:  # Bear
        return risk_parity_weights(cov)
    else:              # Crisis
        return min_variance_weights(cov)


# ─── Backtest Engine ─────────────────────────────────────────────────────────

def run_backtest(feat_df, regimes, asset_returns, lookback=252, tx_cost_bps=7):
    """
    Runs the regime-driven backtest.

    Parameters:
        feat_df       : feature DataFrame (used for its index/alignment)
        regimes       : Series of out-of-sample regime labels (0=Bull,1=Bear,2=Crisis)
        asset_returns : DataFrame of daily log returns for assets (SPY, TLT, GLD)
        lookback      : trailing window (days) for covariance/mean estimation
        tx_cost_bps   : transaction cost in basis points per rebalance

    Returns:
        dict containing daily returns for:
            - strategy_gross  (dynamic, no tx costs)
            - strategy_net    (dynamic, with tx costs)
            - static_6040     (60% SPY / 40% TLT, rebalanced monthly)
            - equal_weight    (1/3 each, rebalanced monthly)
    """
    tx_cost = tx_cost_bps / 10_000

    # Align returns to the regime-labeled dates
    dates      = regimes.index
    rets_slice = asset_returns.loc[dates]

    assets = rets_slice.columns.tolist()
    n      = len(assets)

    strategy_gross = pd.Series(index=dates, dtype=float)
    strategy_net   = pd.Series(index=dates, dtype=float)
    static_6040    = pd.Series(index=dates, dtype=float)
    equal_weight   = pd.Series(index=dates, dtype=float)

    current_w  = np.ones(n) / n    # start equal-weighted
    prev_regime = None
    static_w   = np.array([0.6, 0.3, 0.1])   # SPY 60 / TLT 30 / GLD 10

    for i, date in enumerate(dates):
        regime  = int(regimes.loc[date])
        day_ret = rets_slice.loc[date].values

        # --- Only reoptimize when the regime changes ---
        if regime != prev_regime:
            hist = asset_returns.loc[:date].iloc[-lookback:]
            if len(hist) < 30:
                mu_est  = pd.Series(np.zeros(n), index=assets)
                cov_est = pd.DataFrame(np.eye(n) * 0.01, index=assets, columns=assets)
            else:
                mu_est  = hist.mean() * 252   # annualized mean return
                cov_est = hist.cov()   * 252  # annualized covariance

            new_w    = get_regime_weights(regime, mu_est, cov_est)
            turnover = np.abs(new_w - current_w).sum()
            cost     = turnover * tx_cost
            prev_regime = regime
        else:
            new_w    = current_w
            cost     = 0.0

        # --- Strategy return ---
        gross_ret = new_w @ day_ret

        strategy_gross.loc[date] = gross_ret
        strategy_net.loc[date]   = gross_ret - cost
        current_w                = new_w.copy()

        # --- Static benchmarks (no rebalancing cost modeled for simplicity) ---
        static_6040.loc[date]  = static_w        @ day_ret
        equal_weight.loc[date] = (np.ones(n)/n)  @ day_ret

    return {
        "strategy_gross": strategy_gross,
        "strategy_net":   strategy_net,
        "static_6040":    static_6040,
        "equal_weight":   equal_weight,
    }


def run_backtest_probabilities(feat_df, regime_probs, asset_returns, lookback=252, tx_cost_bps=7, alpha_smooth=0.05):
    """
    Runs a blended portfolio backtest using daily regime probabilities.
    Smooths target weights using exponential smoothing to manage transaction costs.
    """
    tx_cost = tx_cost_bps / 10_000
    dates = regime_probs.index
    rets_slice = asset_returns.loc[dates]

    assets = rets_slice.columns.tolist()
    n = len(assets)

    strategy_gross = pd.Series(index=dates, dtype=float)
    strategy_net = pd.Series(index=dates, dtype=float)
    static_6040 = pd.Series(index=dates, dtype=float)
    equal_weight = pd.Series(index=dates, dtype=float)

    # Initial state
    current_w = np.ones(n) / n
    static_w = np.array([0.6, 0.3, 0.1])

    # Cache optimized candidate weights to save compute time (only recalculate every 5 days)
    cached_w_bull = np.array([1/3, 1/3, 1/3])
    cached_w_bear = np.array([1/3, 1/3, 1/3])
    cached_w_crisis = np.array([1/3, 1/3, 1/3])

    print("Running blended probability backtest...")

    for i, date in enumerate(dates):
        day_ret = rets_slice.loc[date].values
        probs = regime_probs.loc[date].values  # Bull, Bear, Crisis probabilities

        # Re-estimate and optimize every 5 trading days to speed up execution
        if i % 5 == 0 or i == 0:
            hist = asset_returns.loc[:date].iloc[-lookback:]
            if len(hist) < 30:
                mu_est = pd.Series(np.zeros(n), index=assets)
                cov_est = pd.DataFrame(np.eye(n) * 0.01, index=assets, columns=assets)
            else:
                mu_est = hist.mean() * 252
                cov_est = hist.cov() * 252

            # Optimize portfolios for each candidate state
            cached_w_bull = max_sharpe_weights(mu_est, cov_est)
            cached_w_bear = risk_parity_weights(cov_est)
            cached_w_crisis = min_variance_weights(cov_est)

        # Blend targets using state probabilities
        target_w = (probs[0] * cached_w_bull + 
                    probs[1] * cached_w_bear + 
                    probs[2] * cached_w_crisis)
        
        # Normalize target weights to ensure they sum to exactly 1.0
        target_w = np.maximum(target_w, 0)
        target_w /= target_w.sum()

        # Apply exponential smoothing to obtain smooth trade execution weights
        new_w = alpha_smooth * target_w + (1.0 - alpha_smooth) * current_w
        new_w /= new_w.sum()

        # Strategy return calculation
        gross_ret = new_w @ day_ret
        turnover = np.abs(new_w - current_w).sum()
        cost = turnover * tx_cost

        strategy_gross.loc[date] = gross_ret
        strategy_net.loc[date] = gross_ret - cost
        current_w = new_w.copy()

        # Benchmarks
        static_6040.loc[date] = static_w @ day_ret
        equal_weight.loc[date] = (np.ones(n) / n) @ day_ret

    return {
        "strategy_gross": strategy_gross,
        "strategy_net": strategy_net,
        "static_6040": static_6040,
        "equal_weight": equal_weight,
    }




# ─── Performance Metrics ─────────────────────────────────────────────────────

def compute_metrics(returns_series, rf_daily=0.0, label="Strategy"):
    r   = returns_series.dropna()
    ann = r.mean() * 252
    vol = r.std()  * np.sqrt(252)

    sharpe   = (r.mean() - rf_daily) / r.std() * np.sqrt(252) if r.std() > 0 else 0
    downside = r[r < 0].std() * np.sqrt(252)
    sortino  = ann / downside if downside > 0 else 0

    cum          = np.exp(r.cumsum())
    running_max  = cum.cummax()
    drawdown     = (cum / running_max - 1)
    max_dd       = drawdown.min()
    calmar       = ann / abs(max_dd) if max_dd != 0 else 0

    return {
        "Label":        label,
        "Ann. Return":  f"{ann*100:.2f}%",
        "Ann. Vol":     f"{vol*100:.2f}%",
        "Sharpe":       f"{sharpe:.2f}",
        "Sortino":      f"{sortino:.2f}",
        "Max Drawdown": f"{max_dd*100:.2f}%",
        "Calmar":       f"{calmar:.2f}",
    }


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_equity_curves(results, feat_df, regimes, output_path="equity_curve.png"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    colors = {"strategy_net": "#2196F3", "static_6040": "#FF9800", "equal_weight": "#9E9E9E"}
    labels = {"strategy_net": "Dynamic (regime-driven, net of costs)",
              "static_6040":  "Static 60/40",
              "equal_weight": "Equal Weight 1/3"}

    for key, color in colors.items():
        r   = results[key].dropna()
        cum = np.exp(r.cumsum())
        ax1.plot(cum.index, cum.values, color=color, lw=1.5, label=labels[key])

    # Regime shading on top chart
    # If regimes is a Series (hard states), do normal color shading.
    # If regimes is a DataFrame (probabilities), do custom probability-blended shading.
    dates = regimes.index
    if isinstance(regimes, pd.DataFrame):
        # We blend colors: Bull=Green, Bear=Orange, Crisis=Red
        # Normalized RGB values
        c_bull = np.array([165, 214, 167]) / 255.0
        c_bear = np.array([255, 224, 130]) / 255.0
        c_crit = np.array([239, 154, 154]) / 255.0

        for i in range(len(dates) - 1):
            probs = regimes.iloc[i].values # Bull, Bear, Crisis
            # Blended RGB
            blend_color = probs[0] * c_bull + probs[1] * c_bear + probs[2] * c_crit
            ax1.axvspan(dates[i], dates[i+1], color=blend_color, alpha=0.18, linewidth=0)
    else:
        regime_colors = {0: "#a5d6a7", 1: "#ffe082", 2: "#ef9a9a"}
        for i in range(len(dates) - 1):
            state = int(regimes.iloc[i])
            ax1.axvspan(dates[i], dates[i+1], color=regime_colors[state], alpha=0.15)

    ax1.set_title("Regime-Driven TAA Engine vs Static Benchmarks", fontsize=15, fontweight="bold")
    ax1.set_ylabel("Cumulative Return (log-scale)", fontsize=11)
    ax1.set_yscale("log")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Drawdown chart
    r_strat   = results["strategy_net"].dropna()
    cum_strat = np.exp(r_strat.cumsum())
    dd        = cum_strat / cum_strat.cummax() - 1
    ax2.fill_between(dd.index, dd.values, 0, color="#2196F3", alpha=0.4, label="Strategy drawdown")
    ax2.set_ylabel("Drawdown", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="lower left", fontsize=9)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Equity curve saved -> {output_path}")


if __name__ == "__main__":
    from data_pipeline import load_data
    from features import compute_features
    from validation import run_walk_forward

    df      = load_data()
    feat_df = compute_features(df)

    feature_cols  = ["mom_21d_zscore", "vol_21d_zscore"]
    asset_returns = df[["SPY_ret", "TLT_ret", "GLD_ret"]].rename(
        columns={"SPY_ret": "SPY", "TLT_ret": "TLT", "GLD_ret": "GLD"}
    )

    regimes = run_walk_forward(feat_df, feature_cols)
    results = run_backtest(feat_df, regimes, asset_returns)

    # Print metrics table
    strategies = {
        "Dynamic (gross)":     results["strategy_gross"],
        "Dynamic (net 7bps)":  results["strategy_net"],
        "Static 60/40":        results["static_6040"],
        "Equal Weight":        results["equal_weight"],
    }

    print("\n" + "="*75)
    print(f"{'Strategy':<25} {'Ann.Ret':>8} {'Vol':>7} {'Sharpe':>7} "
          f"{'Sortino':>8} {'MaxDD':>8} {'Calmar':>7}")
    print("="*75)
    for label, ret_series in strategies.items():
        m = compute_metrics(ret_series, label=label)
        print(f"{m['Label']:<25} {m['Ann. Return']:>8} {m['Ann. Vol']:>7} "
              f"{m['Sharpe']:>7} {m['Sortino']:>8} {m['Max Drawdown']:>8} {m['Calmar']:>7}")
    print("="*75)

    plot_equity_curves(results, feat_df, regimes)
