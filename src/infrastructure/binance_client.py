from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import requests

from ..application.interfaces import IMarketDataProvider
from ..domain.errors import DataFetchError
from ..domain.models import Candle

logger = logging.getLogger(__name__)


class BinanceDataProvider(IMarketDataProvider):
    """
    Fetches public BTCUSDT 1h klines from Binance's geo-unblocked data API.
    Supports pagination to fetch more than 1000 bars.
    """

    BASE_URL = "https://data-api.binance.vision/api/v3/klines"
    MAX_PER_REQUEST = 1000

    def fetch_historical_klines(self, limit: int = 1000) -> list[Candle]:
        """
        Fetches the latest `limit` hourly BTCUSDT klines.
        Automatically paginates backward when limit > 1000.
        """
        all_candles: list[Candle] = []
        remaining = limit
        end_time: int | None = None

        while remaining > 0:
            batch_size = min(remaining, self.MAX_PER_REQUEST)
            params: dict = {
                "symbol": "BTCUSDT",
                "interval": "1h",
                "limit": batch_size,
            }
            if end_time is not None:
                params["endTime"] = end_time

            batch = self._fetch_single_batch(params)
            if not batch:
                break

            all_candles = batch + all_candles
            remaining -= len(batch)

            # Set end_time to 1ms before the earliest candle in this batch
            # to fetch the previous page
            earliest_timestamp_ms = int(batch[0].timestamp.timestamp() * 1000)
            end_time = earliest_timestamp_ms - 1

            # Rate-limit courtesy
            if remaining > 0:
                time.sleep(0.1)

        return all_candles[-limit:]

    def _fetch_single_batch(self, params: dict) -> list[Candle]:
        """Fetches a single batch of klines from the API."""
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=15)
            response.raise_for_status()
            raw_klines = response.json()
        except requests.RequestException as exc:
            raise DataFetchError(str(exc))
        except ValueError as exc:
            raise DataFetchError(f"Malformed JSON from Binance: {exc}")

        candles: list[Candle] = []
        for row in raw_klines:
            open_time_ms = int(row[0])
            candles.append(
                Candle(
                    timestamp=datetime.fromtimestamp(
                        open_time_ms / 1000.0, tz=timezone.utc
                    ),
                    open_price=float(row[1]),
                    high_price=float(row[2]),
                    low_price=float(row[3]),
                    close_price=float(row[4]),
                    volume=float(row[5]),
                )
            )
        return candles
