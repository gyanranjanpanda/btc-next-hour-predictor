# Task: AlphaI x Polaris Build Challenge - Bitcoin Next Hour Predictor

## Objective
Build a system to predict the price range where Bitcoin (BTC) will land one hour from now using a Geometric Brownian Motion (GBM) simulator. The system consists of a backtesting script (Part A) and a live dashboard (Part B).

## Context
The project is a submission for the AlphaI x Polaris Build Challenge. The goal is to accurately predict the 95% confidence interval for the next hour's Bitcoin price.

## Requirements
1. **Part A — 30-day backtest**:
   - Fetch the last 30 days of BTCUSDT 1-hour bars from Binance (~720 bars) using `https://data-api.binance.vision/api/v3/klines`.
   - Implement a strictly out-of-sample prediction loop ("No peeking").
   - Use the provided GBM simulator logic (using a Student-t distribution) to predict the 95% range for each next bar.
   - Output predictions to `backtest_results.jsonl`.
   - Calculate Coverage (target ~0.95), Average Width, and Winkler score.
2. **Part B — Live Dashboard**:
   - Build a live web dashboard (e.g., using Streamlit, FastAPI + Plotly, or Next.js + Python backend).
   - Fetch the latest closed BTCUSDT 1-hour bar in real time.
   - Run the model on the last 500 bars.
   - Display: Current BTC price, predicted 95% range for the next hour, a chart of the last 50 bars with the predicted range shaded, and the backtest metrics.
3. **Part C — Persistence (Optional Bonus)**:
   - Store historical predictions in a database or file so the dashboard displays a growing timeline of past predictions vs actuals.

## Constraints
- Must not peek ahead in the backtest.
- Must handle volatility clustering appropriately (already partially handled by the starter logic).
- Must use heavy tails (Student-t) rather than normal distributions.
