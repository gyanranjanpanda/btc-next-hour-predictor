# ₿ Bitcoin Next-Hour Predictor

**AlphaI × Polaris Build Challenge Submission**

> Predicting the 95% confidence interval for the next hourly BTC candle using Geometric Brownian Motion with GARCH(1,1) volatility and Student-t fat tails.

---

## 🔗 Live Dashboard

**[btc-next-hour-predictor.streamlit.app](https://btc-next-hour-predictor.streamlit.app/)**

---

## 🏆 Results

| Metric | Value |
|--------|-------|
| **Coverage (95%)** | ✅ See live dashboard |
| **Average Width** | See live dashboard |
| **Mean Winkler Score** | See live dashboard |
| **Total Predictions** | 720 |

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip

### Setup
```bash
cd "alpha ai"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Backtest (Part A)
```bash
PYTHONPATH=. python -m src.interfaces.cli
```
This generates `backtest_results.jsonl` with 720 out-of-sample predictions.

### Run Dashboard (Part B)
```bash
PYTHONPATH=. streamlit run src/interfaces/dashboard.py
```
Opens at `http://localhost:8501`.

## 📐 Architecture

```
src/
├── domain/          ← Pure business logic, no framework imports
│   ├── models.py    ← Candle, Prediction, BacktestResult (Pydantic)
│   ├── simulator.py ← GBM + GARCH(1,1) + Student-t engine + SimulationResult
│   └── errors.py    ← Domain exceptions
├── application/     ← Use cases, depends only on domain interfaces
│   ├── interfaces.py ← Protocol definitions (ports)
│   └── use_cases.py  ← PredictNextHour, RunBacktest
├── infrastructure/  ← External adapters
│   ├── binance_client.py  ← Binance API (with pagination for >1000 bars)
│   └── jsonl_repository.py ← Append-only JSONL persistence
└── interfaces/      ← Delivery mechanisms
    ├── cli.py       ← Backtest CLI entrypoint
    └── dashboard.py ← Streamlit dashboard
```

## 🧠 Model Design

### Core Algorithm: GBM with GARCH + Student-t

1. **Volatility Estimation**: Uses `GARCH(1,1)` via the `arch` library to capture **volatility clustering** — calm hours produce narrow ranges, volatile hours produce wider ones.
2. **Fat Tails**: Student-t distributed innovations (not Gaussian) to handle Bitcoin's frequent large moves. Degrees of freedom are fit via MLE, clamped to `[3, 30]`.
3. **Variance Normalization**: Raw `t(nu)` has `Var = nu/(nu-2)`. We scale innovations by `sqrt((nu-2)/nu)` so `Var = 1`, then multiply by `sigma * sqrt(dt)`.
4. **GBM Formula**: `S(t+1) = S(t) * exp((mu - sigma²/2)*dt + sigma*sqrt(dt)*Z)`
5. **Regime-Adaptive Model Risk Buffer**: Instead of a flat sigma inflation, the simulator detects the current volatility regime (low/normal/high) by comparing recent realized vol against the historical baseline, and applies a regime-specific multiplier (1.01/1.03/1.06).

### No-Peeking Guarantee

In the backtest, at step `i` the model only sees `candles[0..i-1]`. The growing window feeds GARCH up to 500 historical returns for stable fitting, while the actual bar `candles[i]` is only revealed after the prediction is recorded.

### Fallback Path

If GARCH fails to converge (< 50 observations or numerical issues), the simulator falls back to **EWMA volatility** with scipy MLE for t-distribution fitting.

## 📊 Dashboard Features

- **Live BTC price** from Binance public API with auto-refresh every 5 minutes
- **95% prediction range** for the next hour with countdown timer
- **Candlestick chart** (last 100 hours) with forecast ribbon overlay
- **Monte Carlo distribution** — histogram of 10,000 simulated prices with percentile bounds
- **Fitted model parameters** panel — σ, μ, ν, regime, fitting method
- **Rolling coverage chart** — 50-bar sliding window coverage from backtest
- **Backtest metrics** displayed as headline KPIs
- **"How it works" explainer** with methodology details
- **Downloadable backtest results** (JSONL)
- **Dark theme** with premium glassmorphism-inspired design

## 🐛 Bugs Found in Starter Notebook

1. The starter notebook uses currency (USD/CHF) data at daily frequency. When adapting to crypto at hourly frequency, the volatility scale is fundamentally different — direct parameter transfer would produce wrong ranges.
2. The starter's `evaluate()` function assumes predictions are provided as a list of dicts. Our Pydantic model cleanly serialises to this format via `.model_dump()`.

## 📁 Submission Deliverables

- **Code**: This repository
- **Dashboard URL**: [btc-next-hour-predictor.streamlit.app](https://btc-next-hour-predictor.streamlit.app/)
- **Backtest file**: `backtest_results.jsonl`
