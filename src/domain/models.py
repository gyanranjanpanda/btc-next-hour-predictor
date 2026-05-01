from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, ConfigDict


class Candle(BaseModel):
    """Represents a single OHLCV price bar from the exchange."""

    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float

    model_config = ConfigDict(frozen=True)


class Prediction(BaseModel):
    """A predicted price range for a future bar."""

    timestamp: datetime
    lower_bound: float
    upper_bound: float
    actual_close: float | None = None
    confidence_interval: float = 0.95

    model_config = ConfigDict(frozen=True)

    @property
    def width(self) -> float:
        return self.upper_bound - self.lower_bound

    @property
    def contains_actual(self) -> bool:
        if self.actual_close is None:
            return False
        return self.lower_bound <= self.actual_close <= self.upper_bound

    @property
    def winkler_score(self) -> float | None:
        """
        Winkler interval score: rewards narrow ranges, penalizes misses.
        Lower is better. A hit scores its width; a miss adds a penalty
        proportional to the distance the actual price fell outside the range.
        """
        if self.actual_close is None:
            return None

        alpha = 1.0 - self.confidence_interval
        width = self.width

        if self.actual_close < self.lower_bound:
            return width + (2.0 / alpha) * (self.lower_bound - self.actual_close)
        elif self.actual_close > self.upper_bound:
            return width + (2.0 / alpha) * (self.actual_close - self.upper_bound)
        return width


class BacktestResult(BaseModel):
    """Aggregated metrics from a backtest run."""

    total_predictions: int
    coverage: float
    average_width: float
    mean_winkler_score: float

    model_config = ConfigDict(frozen=True)
