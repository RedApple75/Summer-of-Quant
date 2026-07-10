# Regime-Shift: Macro-Aware Tactical Asset Allocation Engine

A system that detects whether the market is in a Bull, Bear, or Crisis state using a Hidden Markov Model, then automatically rebalances a multi-asset portfolio using convex optimization to match that regime — tested under walk-forward validation to ensure results are not a product of hindsight.

---

## Key Decisions

### Why 3 regimes and not 2 or 4?
Three regimes map directly to three meaningfully different portfolio postures: a Bull regime where you want to maximize Sharpe (take equity risk), a Bear regime where you want risk parity (spread the bet), and a Crisis regime where you want to minimize volatility (flee to safety). Collapsing Bear and Crisis into one state would force a single set of portfolio weights onto both a slow drawdown and a violent crash — very different situations. Going to 4+ states creates a statistical problem: with ~2,000 trading days and 3 states you get ~600–700 days per state, which is enough to estimate reliable emission distributions. A 4th state would be chronically data-starved and would likely just be noise.

### Why these specific assets?
The PDF specifies NSE data but this implementation uses `SPY` (US equities), `TLT` (US Treasuries), and `GLD` (gold) via yfinance. This was a deliberate tradeoff: Indian ETF data on yfinance has significantly more gaps, corporate action noise, and calendar mismatches than US instruments, which would silently corrupt the HMM emission estimates. The regime dynamics (Bull/Bear/Crisis) are globally correlated, and SPY/TLT/GLD are liquid, clean proxies that allow us to build the methodology correctly. `^VIX` is used instead of India VIX for the same reason.

### Why diagonal covariance for the HMM?
We chose `covariance_type="diag"` over `"full"`. With 7 input features and 3 states, a full covariance model estimates 3 × (7×8/2) = 84 covariance parameters. Diagonal cuts this to 3 × 7 = 21. On ~2,000 samples this matters: a full covariance matrix with this data-to-parameter ratio will overfit and produce unstable regime assignments that change dramatically with small shifts in the training window — exactly what you do not want in a walk-forward harness.

### Why these feature windows?
Momentum at 5d (weekly direction), 21d (monthly trend), 63d (quarterly trend), and 126d (6-month secular trend) — different windows pick up different persistence timescales, and using all four gives the HMM overlapping but non-redundant views of the same market. Volatility at 5d, 21d, and 63d only — we deliberately omit 126d volatility because at that horizon it largely recapitulates the momentum signal and would add noise to the emission parameters without adding discriminative power.

### Why `min_periods=63` for the expanding Z-score?
Below 63 observations (~3 months of trading days), the sample standard deviation is too noisy for the z-score to carry any real signal — you're essentially dividing by a number with a confidence interval wider than the signal itself. 252 (one full year) would waste too much data in early walk-forward training windows. 63 is the crossover point where the expanding std dev stabilizes enough to be useful.

### Why expanding walk-forward over rolling?
An expanding window (training always starts at the beginning, the end moves forward) rather than a fixed rolling window. HMMs benefit from more data because older regimes still inform transition probability estimates. A rolling window would discard the 2008–2012 experience when training on 2020, which would systematically underestimate the Crisis state parameters and make the model blind to the very events that define what a crisis looks like.

---

## How to Run the Code

### 1. Set up the environment
```bash
python -m venv .venv
.\.venv\Scripts\activate        # Windows
# source .venv/bin/activate    # macOS / Linux
pip install yfinance pandas numpy matplotlib scipy hmmlearn cvxpy
```

### 2. Run the full pipeline
```bash
python main.py
```
This runs the entire pipeline end-to-end: data → features → regime detection → optimization → backtest → results.

### 3. Run individual modules
```bash
python data_pipeline.py       # Verify data ingestion and alignment
python features.py            # Verify feature engineering and z-scoring
python regime_classifier.py   # Fit HMM and generate regime overlay plot
python validation.py          # Run walk-forward harness
python backtest.py            # Run full backtest and print performance summary
```

---

## How to Reproduce the Results

All results are deterministic given the following fixed parameters:

| Parameter | Value | Why |
|---|---|---|
| Data range | `2015-01-01` to `2024-01-01` | Covers 2 full crisis events (2020, 2022) |
| HMM `random_state` | `42` | Fixes random initialization for reproducibility |
| HMM `covariance_type` | `"diag"` | Fewer params, more stable on this sample size |
| HMM `n_components` | `3` | Bull / Bear / Crisis |
| Z-score `min_periods` | `63` | ~3 months minimum lookback |
| Transaction cost | `7 bps` | Mid-point of the 5–10 bps range in the spec |
| Walk-forward test window | `63 days` | ~1 quarter per fold |
| Walk-forward min train | `252 days` | Minimum 1 year of training history |

To reproduce: clone the repo, set up the environment as above, and run `python main.py`. No external data files are needed — all data is fetched live from yfinance on the specified date range.
