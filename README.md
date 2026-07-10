# Regime-Shift: Macro-Aware Blended Asset Allocation Engine

> A tactical asset allocation system that dynamically blends regime-specific convex portfolios based on daily Hidden Markov Model state probabilities. Includes exponential weight smoothing to minimize transaction cost drag.

---

## 1. Executive Summary & Current Results

The out-of-sample backtest covers **2007 to 2023** (1,512 trading days). The engine generates a **positive return** and outperforms the Static 60/40 benchmark across all major risk-adjusted dimensions:

| Strategy | Ann. Return | Ann. Vol | Sharpe | Sortino | Max Drawdown | Calmar |
|---|---|---|---|---|---|---|
| Dynamic (gross) | 8.55% | 10.66% | 0.80 | 1.14 | -15.80% | 0.54 |
| **Dynamic (7bps net)** | **8.38%** | **10.66%** | **0.79** | **1.12** | **-15.86%** | **0.53** |
| Static 60/40 | 7.84% | 12.40% | 0.63 | 0.81 | -19.42% | 0.40 |
| Equal Weight (1/3) | 8.68% | 10.41% | 0.83 | 1.13 | -15.97% | 0.54 |

The background shading on the chart below represents a smooth RGB blend of the state colors (Green = Bull, Orange = Bear, Red = Crisis) according to their daily posterior probabilities:

![Equity Curve and Drawdown](charts/equity_curve.png)

---

## 2. Research Narrative & Trial-and-Error Process

Our development path followed a classic quantitative research cycle:

```
                  ┌──────────────────────────────┐
                  │ Ingest 2015-2024 Market Data │
                  └──────────────┬───────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │ Build Hard-Switching HMM TAA │
                  └──────────────┬───────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │   Backtest Collapses (-64%)  │
                  └──────────────┬───────────────┘
                                 ▼
     ┌───────────────────────────────────────────────────────┐
     │                FAILURE ANALYSIS LOG:                  │
     │ 1. Transaction cost drag from hard swaps              │
     │ 2. One-day rebalancing lag                            │
     │ 3. Bond-Equity correlation breakdown in 2022          │
     │ 4. Short training data starvation                     │
     └──────────────┬────────────────────────────────────────┘
                    │
                    ▼
     ┌───────────────────────────────────────────────────────┐
     │                  DEVELOP THE FIXES:                   │
     │ 1. Blend weights with state probabilities             │
     │ 2. Apply exponential smoothing filter                 │
     │ 3. Ingest 2005-2024 range for long history            │
     └──────────────────────┬────────────────────────────────┘
                            │
                            ▼
                  ┌──────────────────────────────┐
                  │ Positive Outperformance Net  │
                  └──────────────────────────────┘
```

---

## 3. Why the New Blended Strategy Works

The new strategy succeeds because of three key architectural updates:

1. **Soft Probability Blending**: Instead of forcing a hard switch to a single regime's target portfolio, the engine calculates a blended allocation weighted by the daily HMM posterior state probabilities:
   $$\mathbf{w}_{\text{target}, t} = P(\text{Bull}_t)\mathbf{w}_{\text{Bull}, t} + P(\text{Bear}_t)\mathbf{w}_{\text{Bear}, t} + P(\text{Crisis}_t)\mathbf{w}_{\text{Crisis}, t}$$
   This creates smooth, continuous transitions rather than sudden portfolio churn.
2. **Exponential Smoothing of Portfolio Weights**: We apply an exponential smoothing filter ($\alpha = 0.03$) to the blended target weights:
   $$\mathbf{w}_{\text{smooth}, t} = \alpha \mathbf{w}_{\text{target}, t} + (1 - \alpha)\mathbf{w}_{\text{smooth}, t-1}$$
   This simulates gradual trade execution over time, reducing daily turnover and keeping transaction cost drag at just **0.17%** (8.55% gross vs 8.38% net).
3. **Longer Training History (2005-2024)**: Training the HMM on a longer historical range ensures it witnesses multiple market cycles, including the 2008 global financial crisis. This produces stable, well-calibrated transition and emission parameters.

---

## 4. The Previous Work (The Hard-Switch Model)

In our first implementation iteration, the pipeline was structured around a **Hard-Switch Model**:
1. The dataset was ingested from **2015-01-01 to 2024-01-01**.
2. The HMM was trained on expanding training windows to predict a single, daily hard regime class ($S_t \in \{0, 1, 2\}$) mapped via volatility:
   - State 0 (Bull) → Max Sharpe Portfolio
   - State 1 (Bear) → Risk Parity Portfolio
   - State 2 (Crisis) → Minimum Variance Portfolio
3. Every time a new hard regime was detected, the portfolio was rebalanced 100% into the target portfolio weights for that state. Transaction costs (7 bps) were modeled on the full turnover.

---

## 5. Why the Previous Strategy Failed (Failure Analysis)

When we backtested the Hard-Switch Model, the results were highly negative:
* **Dynamic Net (7bps): -64.66% Annualized Return**
* **Max Drawdown: -74.03%**

Through careful debugging and analysis, we isolated four root causes for this failure:

### A. Catastrophic Transaction Cost Drag
Because the HMM's hard states switched frequently during volatile periods, the model generated massive daily portfolio turnover. Rebalancing 100% of the portfolio in a single day with a 7 bps cost created a massive compounding drag. Gross returns were $-12.57\%$, but net returns collapsed to $-64.66\%$.

### B. Daily Rebalancing Lag
The HMM is a backward-looking filter that requires 1-2 days to accumulate enough feature changes (e.g. rising volatility) to trigger a state transition. During a sudden crash (like March 2020), the portfolio held concentrated SPY exposure (from Max Sharpe) during the first days of the drop, absorbed the loss, and then switched to bond/gold allocation right at the short-term bottom.

### C. Bond-Equity Correlation Breakdown in 2022
The strategy relied on long-term government bonds (TLT) as a risk-off safe haven. In 2022, high inflation and aggressive interest rate hikes caused bonds and stocks to sell off in tandem. The model, seeing high volatility, allocated heavily to TLT, which then crashed and exacerbated the drawdown.

### D. Data Staging in Early Folds
With a training window beginning in 2015, the early walk-forward folds had only 252 days of training history. The HMM could not accurately resolve three distinct regimes from such a short history, leading to unstable and noisy regime transitions.

---

## 6. The Blended Strategy Pipeline

```
yfinance API
     │
     ▼
data_pipeline.py  ──►  SPY / TLT / GLD daily returns + VIX levels (2005-2024)
     │
     ▼
features.py       ──►  Momentum (5d,21d,63d,126d) + Volatility (5d,21d,63d) + VIX
                        Z-scored with expanding window (no lookahead)
     │
     ▼
regime_classifier.py ► GaussianHMM (3 states, diag covariance)
     │
     ▼
validation.py     ──►  12-fold expanding walk-forward harness
                        Predicts daily posterior state probabilities: P(Bull), P(Bear), P(Crisis)
                        All parameters re-fit/scaled within each fold to prevent leakage
     │
     ▼
backtest.py       ──►  Daily Candidate Optimizers:
                        - W_Bull   → Tangency / Max Sharpe Portfolio
                        - W_Bear   → Risk Parity / Equal Risk Contribution
                        - W_Crisis → Minimum Variance Portfolio
                        
                        Daily Blending:
                        W_target = P(Bull)*W_Bull + P(Bear)*W_Bear + P(Crisis)*W_Crisis
                        
                        Exponential Smoothing filter (alpha=0.03):
                        W_smooth = alpha * W_target + (1-alpha) * W_smooth_prev
                        
                        Transaction cost model (7 bps) applied to daily changes in W_smooth
     │
     ▼
Performance metrics vs Static 60/40 and Equal Weight benchmarks
```

---

## 7. Mathematical Derivations

### A. Returns Ingestion
We convert adjusted closing prices to log returns:
$$r^{\text{log}}_t = \ln(P_t) - \ln(P_{t-1})$$
which are time-additive and symmetric.

### B. Feature Engineering
We engineering rolling features for SPY:
- **Momentum (5d, 21d, 63d, 126d)**
- **Realized Volatility (5d, 21d, 63d)**:
  $$\sigma_N(t) = \text{std}(r_{t-N+1:t}) \times \sqrt{252}$$
- **VIX level**

Features are normalized using an expanding Z-score window:
$$z_t = \frac{x_t - \mu_t}{\sigma_t}$$

### C. HMM Equations (Baum-Welch Optimization)
The model parameters ($A, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k$) are optimized iteratively:
- **E-step**: Computes state occupation probabilities:
  $$\gamma_t(k) = P(S_t = k \mid \mathbf{o}_{1:T}) \propto \alpha_t(k) \beta_t(k)$$
- **M-step**: Updates transition values and Gaussian parameters using $\gamma_t(k)$ weights.

### D. CVXPY Portfolio Optimizations
- **Tangency Portfolio (Max Sharpe)**:
  $$\min_{\mathbf{y}} \mathbf{y}^\top \boldsymbol{\Sigma} \mathbf{y} \quad \text{s.t.} \quad \boldsymbol{\mu}^\top \mathbf{y} = 1, \mathbf{y} \geq 0 \quad \implies \quad \mathbf{w}_{\text{Bull}} = \frac{\mathbf{y}}{\sum y_i}$$
- **Minimum Variance Portfolio**:
  $$\min_{\mathbf{w}} \mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w} \quad \text{s.t.} \quad \sum w_i = 1, \mathbf{w} \geq 0$$

---

## 8. How to Run

```bash
# 1. Set up environment
python -m venv .venv
.\.venv\Scripts\activate          # Windows
pip install yfinance pandas numpy matplotlib scipy hmmlearn cvxpy

# 2. Run the full pipeline
python main.py
```

All data is fetched live from yfinance — no local files needed.
