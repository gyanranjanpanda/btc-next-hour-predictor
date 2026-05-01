from __future__ import annotations

from typing import Protocol

from ..domain.models import Candle, Prediction


class IMarketDataProvider(Protocol):
    """Port for fetching OHLCV candle data from any exchange."""

    def fetch_historical_klines(self, limit: int = 1000) -> list[Candle]: ...


class IPredictionRepository(Protocol):
    """Port for persisting and retrieving historical predictions."""

    def save(self, prediction: Prediction) -> None: ...

    def get_all(self) -> list[Prediction]: ...
