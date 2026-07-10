# Regime-Shift: Macro-Aware Tactical Asset Allocation Engine

This project builds a **Regime-Shift: Macro-Aware Tactical Asset Allocation Engine** using a Hidden Markov Model (HMM) to classify market states and a convex optimizer (`cvxpy`) to adjust portfolio weights dynamically across stocks (`SPY`), bonds (`TLT`), and gold (`GLD`).

## Project Deliverables Status

- [x] **1. Data Ingestion Pipeline**: Fetches daily prices from `yfinance` for `SPY`, `TLT`, `GLD`, and `^VIX`. Aligns datasets and computes daily log returns.
- [x] **2. Feature Engineering**: Calculates multi-horizon rolling momentum (5d, 21d, 63d, 126d) and rolling realized volatility (5d, 21d, 63d). Standardizes features using expanding-window Z-scores to strictly prevent lookahead bias.
- [ ] **3. HMM Regime Classifier**: Detects 3 hidden states (Bull, Bear, Crisis) using `hmmlearn` and maps them by volatility. Plots regimes overlaid on price charts.
- [ ] **4. Walk-Forward Validation Harness**: Re-fits the HMM and scales features strictly inside chronological train/test splits.
- [ ] **5. Convex Portfolio Optimizer**: Dynamic weights optimization using `cvxpy` matching the detected regime with transaction costs (5–10 bps per rebalance).
- [ ] **6. Performance Benchmarking**: Evaluates dynamic strategy returns, Sharpe ratio, Sortino ratio, Max Drawdown, and Calmar ratio against static 60/40 and equal-weight portfolios.
- [ ] **7. Replication & Run Scripts**: Main orchestrator script running the entire pipeline top-to-bottom.

---

## Technical Design Decisions

### Why 3 Regimes?
Market dynamics are generally categorized into three macroeconomic phases:
1. **Bull (Low Volatility)**: Characterized by steady upward trending prices and low volatility. Highly supportive of equity risk.
2. **Bear (Medium Volatility)**: Characterized by choppy or declining prices with elevated volatility. Bonds and gold are preferred over equities.
3. **Crisis (High Volatility)**: Characterized by sudden, extreme downward jumps and high market fear (e.g. 2008 Lehman collapse, 2020 COVID crash). Demands capital preservation (flight to gold, bonds, or cash).

### Feature Selection Rationale
- **Momentum** (5d, 21d, 63d, 126d): Captures short-term reversals and medium-to-long-term trends across horizons.
- **Realized Volatility** (5d, 21d, 63d): Standard deviations of returns capture short-term uncertainty spikes.
- **VIX Index Levels**: Serves as a forward-looking expectation of 30-day implied volatility, reflecting institutional fear.

### Preventing Lookahead Bias
Traditional machine learning methods shuffle data, leaking future patterns into the past. We strictly avoid lookahead bias by:
- Scaling features with **expanding windows** (`min_periods=63`) so that a Z-score at time $t$ only uses indices $\le t$.
- Utilizing a **walk-forward validation loop** where the HMM is re-fit inside each training split, rather than once on the full sample.

---

## Setup & Running the Code

### 1. Prerequisites & Environment
Ensure Python 3.9+ is installed. Create a virtual environment and install the dependencies:
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install yfinance pandas numpy matplotlib scipy hmmlearn cvxpy
```

### 2. Verify Data Ingestion & Features
Run the modules to verify that data is loaded and features are engineered without errors:
```bash
python data_pipeline.py
python features.py
```
