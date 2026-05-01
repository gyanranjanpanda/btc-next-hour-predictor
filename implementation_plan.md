# Implementation Plan

## Phase 1: Data Acquisition and Core Modeling (Part A)
1. **Setup Environment**: Initialize a Python project with necessary dependencies (Pandas, NumPy, requests, scipy, streamlit, plotly).
2. **Fetch Historical Data**: Write a module to fetch 1-hour BTCUSDT klines from Binance's public data API (`https://data-api.binance.vision/api/v3/klines`).
3. **Port Simulator Logic**: Adapt the Geometric Brownian Motion (GBM) simulator from the starter Colab notebook to work on Bitcoin data locally.
4. **Implement Backtesting Engine**:
   - Create a rolling window loop over the last ~720 bars.
   - Ensure strictly no forward-looking data leaks.
   - For each step, fit volatility and simulate the next step using Student-t innovations.
   - Evaluate the predicted 95% range against the actual next close.
5. **Calculate Metrics & Export**:
   - Compute coverage, average width, and Winkler score.
   - Save predictions to `backtest_results.jsonl`.

## Phase 2: Live Dashboard (Part B)
1. **Choose Stack**: We will use Streamlit for rapid, robust, data-centric dashboarding as recommended in the challenge description.
2. **Live Data Fetcher**: Implement a function to pull the most recent 500 hourly bars from Binance on-demand.
3. **Live Predictor**: Wire the core modeling function to run on the live dataset.
4. **UI Construction**:
   - Header displaying the backtest metrics.
   - Big number display for current price and predicted range.
   - Plotly chart showing the last 50 bars and a shaded ribbon for the next-hour prediction.

## Phase 3: Persistence (Part C - Optional but Planned)
1. **Setup Storage**: Add a simple local SQLite database or append-only JSONL file for storing live predictions.
2. **Record & Retrieve**: Update the dashboard to record its prediction and fetch past predictions to overlay on the chart.

## Phase 4: Refinement and Review
1. Ensure code adheres to standards (`/Users/mac/.gemini/standards/python.md`).
2. Implement robust error handling for API rate limits and network issues.
3. Prepare the final codebase and documentation for submission.
