# Clean Architecture Design

This document outlines the architectural structure for the AlphaI x Polaris Build Challenge submission, strictly following Clean Architecture principles.

## 1. Domain Layer (`domain/`)
The absolute core of the business logic. It has no dependencies on any external framework, database, or UI.
- **Models**: 
  - `Candle`: Represents a single Bitcoin price bar (open, high, low, close, volume, timestamp).
  - `Prediction`: Represents a predicted range (lower_bound, upper_bound, confidence_interval).
  - `BacktestResult`: Represents the result of a backtest (coverage, average_width, winkler_score).
- **Core Logic**:
  - `GBMSimulator`: The mathematical engine that simulates future prices using Geometric Brownian Motion and Student-t distributions to account for fat tails. It takes an array of recent returns to calculate volatility (accounting for volatility clustering) and outputs a `Prediction`.

## 2. Application Layer (`application/`)
Contains the application specific business rules and Use Cases. It orchestrates the flow of data to and from the domain entities.
- **Use Cases**:
  - `PredictNextHourUseCase`: Orchestrates fetching the latest data from the infrastructure, feeding it to the `GBMSimulator`, and returning a prediction.
  - `RunBacktestUseCase`: Orchestrates the 30-day historical fetch, loops over the data without peeking, generates predictions step-by-step, evaluates them, and returns a `BacktestResult`.
  - `SavePredictionUseCase`: Records a prediction for historical persistence.
  - `GetHistoricalPredictionsUseCase`: Retrieves past predictions for the dashboard timeline.
- **Interfaces (Ports)**:
  - `IMarketDataProvider`: Interface for fetching klines (candles).
  - `IPredictionRepository`: Interface for storing and retrieving prediction history.

## 3. Infrastructure Layer (`infrastructure/`)
Implementations of the interfaces defined in the application layer. This layer interacts with the outside world.
- **Adapters**:
  - `BinanceDataProvider`: Implements `IMarketDataProvider`. Uses `requests` to fetch data from `https://data-api.binance.vision/api/v3/klines`.
  - `JsonlPredictionRepository`: Implements `IPredictionRepository`. Saves predictions to a `.jsonl` file.

## 4. Interfaces/Presentation Layer (`interfaces/`)
The delivery mechanism. In this case, the Streamlit application and CLI scripts.
- **CLI (`cli.py`)**: Script to run the `RunBacktestUseCase` and save `backtest_results.jsonl`.
- **Dashboard (`app.py`)**: The Streamlit application that uses `PredictNextHourUseCase` and `GetHistoricalPredictionsUseCase` to display:
  - The live, premium UI.
  - Big number KPIs.
  - Interactive Plotly charts showing historical prices and shaded prediction ribbons.
