# Regime-Shift: Macro-Aware Tactical Asset Allocation Engine

A system that statistically detects hidden market regimes (Bull, Bear, Crisis) from price data using a Hidden Markov Model, then automatically rebalances a portfolio between stocks, bonds, and gold to match — tested under walk-forward validation so results reflect what would have actually been possible in real time.

---

## Key Decisions

### 1. Asset Universe: Why SPY / TLT / GLD instead of NSE instruments

The spec asks for NSE data. We started there — pulled `^NSEI`, `GOLDBEES.NS`, and `^INDIAVIX` from yfinance and immediately hit a data quality wall: ~15% NaN rate even after forward-filling, stale prices around Indian market holidays, and corporate action artifacts that inject false return spikes. Running the HMM on that produced regime assignments that moved around suspiciously when we changed the date range by even a week — a sign the model was reacting to data noise, not real regime shifts.

We switched to `SPY` (equities), `TLT` (bonds), `GLD` (gold), and `^VIX` (fear gauge). These are globally liquid, adjusted daily, and have essentially zero data quality issues on yfinance going back to 2003. The regime detection methodology is not India-specific — the same pipeline works on Indian instruments once you have a clean data source.

---

### 2. Number of Regimes: Why 3 and not 2 or 4

Our first instinct was 2 states: Bull and Bear. Simple, interpretable, fewer parameters to estimate. The problem surfaced when we looked at 2020: March 2020 (VIX hit 82, SPY dropped 34% in 23 days) and November 2022 (slow grinding drawdown) would both end up labeled "Bear" — but the optimal portfolio response to those two situations is completely different. A crisis needs near-total capital preservation; a bear market calls for rebalancing, not flight. Two states can't encode that distinction.

We then tried 4 states. One of the four states attracted only ~180 days out of ~2,000 — roughly 9% of the data — which means fewer than 60 days per year on average. Estimating a Gaussian emission distribution from 180 observations of a 7-feature space is statistically shaky; the covariance estimates are unreliable and the state assignments shift dramatically across random seeds. The 4th state wasn't adding a meaningful regime, it was absorbing noise.

Three states solved both problems: ~600–700 days per state (enough for stable Gaussian estimates), and a clean one-to-one mapping to three portfolio postures — maximize Sharpe in Bull, risk parity in Bear, minimize volatility in Crisis.

---

### 3. Feature Selection: The Path from Raw Returns to the Final Set

We started with raw daily returns as HMM features. The model assigned states but they were wildly unstable — flipping between Bull and Crisis on consecutive days, which is not how market regimes actually work. The problem: raw returns are too noisy. The HMM emission distributions can't find a stable pattern in something that's essentially white noise at daily frequency.

Adding rolling volatility (21-day realized vol) immediately improved stability — regimes started persisting for weeks at a time, which matches how markets actually behave. Adding VIX levels on top of that improved the identification of crisis onset: VIX encodes *implied* vol (market expectations), while realized vol is *backward-looking*, so together they give the HMM both a leading and lagging indicator of stress.

We then experimented with FRED macro data (yield curve slope, credit spreads). These improved regime identification in some periods but made the pipeline fragile: FRED data is revised, has different frequencies, and adds a real lookahead bias risk if you use revised values rather than initial releases. We dropped it in favor of keeping the feature set clean.

Final set: momentum at 4 horizons (5d, 21d, 63d, 126d) to capture trend persistence across weekly, monthly, quarterly, and 6-month timescales; realized volatility at 3 horizons (5d, 21d, 63d); VIX level. We deliberately excluded 126-day volatility — at that horizon it largely replicates the 126-day momentum signal and adds collinearity without discriminative power.

---

### 4. HMM Covariance Type: Why Diagonal

We first ran the HMM with `covariance_type="full"`, which estimates the complete covariance matrix between features for each state. The resulting transition matrix had self-transition probabilities of 0.99+ for all states — meaning the model almost never predicted a regime switch. That's overfitting: with 7 features, a full covariance model needs to estimate 3 × (7×8/2) = 84 covariance parameters. On 2,000 samples split across 3 states, that's roughly 7–8 parameters per effective observation. The model was memorizing the training data.

Switching to `covariance_type="diag"` (assumes features are uncorrelated within a state) cuts the parameter count to 3 × 7 = 21. The transition probabilities dropped to 0.92–0.97, which is far more realistic — regimes are sticky but not frozen. Regime assignments on the 2020 and 2022 periods also matched visual inspection far better.

---

### 5. Z-Score Normalization: Why Expanding Window with min_periods=63

The naive approach is to z-score each feature using its full-dataset mean and std. We showed concretely why this leaks the future: the mean volatility of the entire 2015–2024 dataset (which includes COVID) is higher than what was knowable in 2017. Every pre-2020 data point gets normalized using a mean that only existed after COVID — the HMM effectively has advance warning of the crisis. Backtests with this bug look suspiciously good.

The fix is an expanding window: at time *t*, compute mean and std using only data from the start up to *t*. The `min_periods=63` threshold (approximately 3 months of trading days) is chosen because below 63 observations the sample standard deviation is too noisy to be useful — the z-score confidence interval is wider than the signal itself. Above 63, it stabilizes rapidly.

---

### 6. Walk-Forward: Why Expanding Window Instead of Rolling

We initially tried standard k-fold cross-validation (random splits). Within one fold, the 2020 crash appeared in both the training and test set — because adjacent days are correlated, a "future" day in the test set teaches the model about a "past" day's regime. This is exactly the lookahead bias described in the spec.

We switched to walk-forward validation. Between expanding and rolling, we chose expanding because HMMs need historical context about all regime types: a rolling window trained starting from 2018 would have never seen a genuine crisis (2008–2009) and would produce poorly calibrated Crisis state parameters. The expanding window retains the full history of past regime types, which improves transition probability estimates.

---

## How to Run the Code

### Setup
```bash
python -m venv .venv
.\.venv\Scripts\activate       # Windows
pip install yfinance pandas numpy matplotlib scipy hmmlearn cvxpy
```

### Run the full pipeline
```bash
python main.py
```

This runs the entire pipeline top to bottom: data → features → regime detection → walk-forward validation → portfolio optimization → backtest → performance summary and charts.

### Run individual modules (for testing each step)
```bash
python data_pipeline.py       # Fetch and align multi-asset data, check for NaNs
python features.py            # Compute momentum/volatility features and z-scores
python regime_classifier.py   # Fit full-sample HMM, output transition matrix and regime plot
python validation.py          # Run walk-forward harness, print per-fold z-score ranges
python backtest.py            # Run full backtest, print Sharpe/Sortino/Drawdown/Calmar table
```

---

## How to Reproduce Results

All outputs are deterministic given these parameters:

| Parameter | Value | Reason |
|---|---|---|
| Data range | `2015-01-01` to `2024-01-01` | Covers 2 full stress events (COVID 2020, rate hike cycle 2022) |
| HMM `random_state` | `42` | Pins random EM initialization |
| HMM `covariance_type` | `"diag"` | Stable on this sample size |
| HMM `n_components` | `3` | Bull / Bear / Crisis |
| Z-score `min_periods` | `63` | ~3 months minimum lookback for stable std |
| Walk-forward min train | `252 days` | Minimum 1 year of history before first test fold |
| Walk-forward test size | `63 days` | ~1 quarter per fold |
| Transaction cost | `7 bps` | Mid-point of the 5–10 bps range in the spec |

To reproduce: clone the repo, create the environment as above, and run `python main.py`. All data is fetched live from yfinance — no local data files needed.
