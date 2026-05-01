from __future__ import annotations

import logging
from datetime import timedelta

from ..domain.models import BacktestResult, Candle, Prediction
from ..domain.simulator import GBMSimulator
from .interfaces import IMarketDataProvider

logger = logging.getLogger(__name__)


class PredictNextHourUseCase:
    """Fetches live data and produces a single next-hour prediction."""

    def __init__(
        self, data_provider: IMarketDataProvider, simulator: GBMSimulator
    ) -> None:
        self.data_provider = data_provider
        self.simulator = simulator

    def execute(self, lookback: int = 500) -> tuple[Prediction, list[Candle]]:
        """
        Returns the prediction for the next hour and the candles used.
        The candle list is returned so the dashboard can render the chart
        without a second API call.
        """
        candles = self.data_provider.fetch_historical_klines(limit=lookback)

        raw_pred = self.simulator.predict_next_candle(candles)
        next_hour = candles[-1].timestamp + timedelta(hours=1)

        prediction = Prediction(
            timestamp=next_hour,
            lower_bound=raw_pred.lower_bound,
            upper_bound=raw_pred.upper_bound,
            confidence_interval=raw_pred.confidence_interval,
        )
        return prediction, candles


class RunBacktestUseCase:
    """
    Runs a strictly out-of-sample walk-forward backtest.
    At each step i, only candles[0..i-1] are visible to the simulator.
    """

    def __init__(
        self, data_provider: IMarketDataProvider, simulator: GBMSimulator
    ) -> None:
        self.data_provider = data_provider
        self.simulator = simulator

    def execute(
        self, lookback_window: int = 50, test_size: int = 720
    ) -> tuple[BacktestResult, list[Prediction]]:
        total_required = lookback_window + test_size
        candles = self.data_provider.fetch_historical_klines(limit=total_required)

        actual_test_size = len(candles) - lookback_window
        if actual_test_size < test_size:
            logger.warning(
                "Fetched %d candles (%d test bars), wanted %d test bars",
                len(candles),
                actual_test_size,
                test_size,
            )

        predictions: list[Prediction] = []

        for step in range(lookback_window, len(candles)):
            history = candles[: step]  # Everything before the target bar
            target_candle = candles[step]

            raw_pred = self.simulator.predict_next_candle(history)

            prediction = Prediction(
                timestamp=target_candle.timestamp,
                lower_bound=raw_pred.lower_bound,
                upper_bound=raw_pred.upper_bound,
                confidence_interval=raw_pred.confidence_interval,
                actual_close=target_candle.close_price,
            )
            predictions.append(prediction)

        return self._aggregate_metrics(predictions), predictions

    @staticmethod
    def _aggregate_metrics(predictions: list[Prediction]) -> BacktestResult:
        if not predictions:
            return BacktestResult(
                total_predictions=0,
                coverage=0.0,
                average_width=0.0,
                mean_winkler_score=0.0,
            )

        hits = sum(1 for p in predictions if p.contains_actual)
        coverage = hits / len(predictions)
        avg_width = sum(p.width for p in predictions) / len(predictions)

        winkler_scores = [
            p.winkler_score for p in predictions if p.winkler_score is not None
        ]
        mean_winkler = (
            sum(winkler_scores) / len(winkler_scores) if winkler_scores else 0.0
        )

        return BacktestResult(
            total_predictions=len(predictions),
            coverage=coverage,
            average_width=avg_width,
            mean_winkler_score=mean_winkler,
        )
