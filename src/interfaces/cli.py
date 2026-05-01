"""
CLI entrypoint for the 30-day backtest (Part A of the challenge).

Run:
    python -m src.interfaces.cli
"""
from __future__ import annotations

import logging
import sys

from src.application.use_cases import RunBacktestUseCase
from src.domain.simulator import GBMSimulator
from src.infrastructure.binance_client import BinanceDataProvider
from src.infrastructure.jsonl_repository import JsonlPredictionRepository

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

BACKTEST_OUTPUT = "backtest_results.jsonl"


def run_backtest() -> None:
    logger.info("Initialising 30-day BTC backtest …")

    data_provider = BinanceDataProvider()
    simulator = GBMSimulator(
        num_simulations=10_000,
        volatility_lookback=24,
        ewma_span=12,
    )
    backtest = RunBacktestUseCase(data_provider, simulator)

    # 720 test bars = 30 days of 1h candles
    # 50 bar lookback = enough history for GARCH to initialise
    lookback_window = 50
    test_bars = 720

    logger.info(
        "Fetching %d bars (%d lookback + %d test) …",
        lookback_window + test_bars,
        lookback_window,
        test_bars,
    )

    try:
        result, predictions = backtest.execute(
            lookback_window=lookback_window,
            test_size=test_bars,
        )
    except Exception as exc:
        logger.error("Backtest failed: %s", exc)
        sys.exit(1)

    logger.info("═══════════════════════════════════════")
    logger.info("  Backtest Complete — %d predictions", result.total_predictions)
    logger.info("  Coverage (95%%):     %.4f", result.coverage)
    logger.info("  Average Width:      $%.2f", result.average_width)
    logger.info("  Mean Winkler Score: %.2f", result.mean_winkler_score)
    logger.info("═══════════════════════════════════════")

    repo = JsonlPredictionRepository(BACKTEST_OUTPUT)
    for prediction in predictions:
        repo.save(prediction)

    logger.info("Saved %d predictions → %s", len(predictions), BACKTEST_OUTPUT)


if __name__ == "__main__":
    run_backtest()
