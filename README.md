# Regime-Shift: Macro-Aware Tactical Asset Allocation Engine

An asset allocation engine that uses a Hidden Markov Model (HMM) to classify market regimes (Bull, Bear, Crisis) and a convex optimizer (`cvxpy`) to adjust portfolio weights dynamically across stocks (`SPY`), bonds (`TLT`), and gold (`GLD`).

---

## 1. Key Decisions

### Why 3 Regimes?
Market dynamics are classified into three distinct macroeconomic regimes:
1. **Bull (Low Volatility)**: Characterized by steady upward trending prices and low volatility. Highly supportive of equity risk.
2. **Bear (Medium Volatility)**: Characterized by choppy or declining prices with elevated volatility. Bonds and gold are preferred over equities.
3. **Crisis (High Volatility)**: Characterized by sudden, extreme downward jumps and high market fear (e.g., 2008 Lehman collapse, 2020 COVID crash). Demands capital preservation (flight to gold, bonds, or cash).

### Why These Features?
- **Momentum** (5-day, 21-day, 63-day, 126-day rolling returns): Captures short-term reversals and medium-to-long-term trends across horizons.
- **Realized Volatility** (5-day, 21-day, 63-day rolling standard deviations): Captures volatility spikes and market stress levels.
- **VIX Index Levels**: Serves as a forward-looking expectation of 30-day implied volatility, reflecting institutional fear.
- **Expanding Z-Score Normalization**: Standardizes features using expanding windows (`min_periods=63`) to prevent lookahead bias, ensuring that the feature normalization at time $t$ uses only information available up to time $t$.

---

## 2. How to Run the Code

### Setup Environment
1. Create a Python 3.9+ virtual environment:
   ```bash
   python -m venv .venv
   ```
2. Activate the virtual environment:
   - On Windows:
     ```bash
     .\.venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
3. Install required dependencies:
   ```bash
   pip install yfinance pandas numpy matplotlib scipy hmmlearn cvxpy
   ```

### Run the Pipeline
Execute the main orchestrator script:
```bash
python main.py
```

---

## 3. How to Reproduce the Results

To reproduce our results exactly, the pipeline runs deterministically using the following controls:
1. **Data Range**: All assets are downloaded from yfinance for the period `2015-01-01` to `2024-01-01`.
2. **Features**: Standardised using expanding Z-score scaling with a minimum of 63 trading days lookback.
3. **Model Seed**: The Hidden Markov Model is fitted with `random_state=42` to ensure reproducible state classification and Viterbi path predictions.
4. **Validation**: Performed via walk-forward validation where the HMM is refit inside chronological train/test splits.
