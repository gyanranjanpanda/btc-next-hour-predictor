"""Tests for the GBM Simulator."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.domain.models import Candle
from src.domain.simulator import GBMSimulator, SimulationResult
from src.domain.errors import SimulationError


def _make_candles(prices: list[float]) -> list[Candle]:
    """Helper: generates candles from a list of close prices."""
    base = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)
    return [
        Candle(
            timestamp=base + timedelta(hours=i),
            open_price=p * 0.999,
            high_price=p * 1.002,
            low_price=p * 0.998,
            close_price=p,
            volume=100.0 + i,
        )
        for i, p in enumerate(prices)
    ]


class TestGBMSimulator:
    @pytest.fixture
    def simulator(self) -> GBMSimulator:
        return GBMSimulator(
            num_simulations=5_000,
            volatility_lookback=24,
            ewma_span=12,
        )

    def test_rejects_insufficient_candles(self, simulator: GBMSimulator) -> None:
        candles = _make_candles([78000.0] * 10)
        with pytest.raises(SimulationError, match="Need >= 25 candles"):
            simulator.predict_next_candle(candles)

    def test_returns_simulation_result(self, simulator: GBMSimulator) -> None:
        """predict_next_candle should return a SimulationResult with metadata."""
        np.random.seed(42)
        prices = [78000.0 + np.random.randn() * 50 for _ in range(60)]
        candles = _make_candles(prices)
        result = simulator.predict_next_candle(candles)

        assert isinstance(result, SimulationResult)
        assert result.prediction.lower_bound < result.prediction.upper_bound
        assert result.fitting_method in ("garch", "ewma")
        assert result.volatility_regime in ("low", "normal", "high")
        assert len(result.simulated_prices) == 5_000
        assert result.fitted_nu >= 3.0
        assert result.fitted_sigma > 0

    def test_prediction_returns_valid_bounds(self, simulator: GBMSimulator) -> None:
        """Prediction bounds should bracket the current price reasonably."""
        np.random.seed(42)
        prices = [78000.0 + np.random.randn() * 50 for _ in range(60)]
        candles = _make_candles(prices)
        result = simulator.predict_next_candle(candles)

        assert result.prediction.lower_bound < result.prediction.upper_bound
        assert result.prediction.lower_bound > 0
        assert result.prediction.confidence_interval == 0.95

    def test_prediction_width_is_positive(self, simulator: GBMSimulator) -> None:
        np.random.seed(123)
        prices = [78000.0 + np.random.randn() * 100 for _ in range(60)]
        candles = _make_candles(prices)
        result = simulator.predict_next_candle(candles)
        assert result.prediction.width > 0

    def test_volatile_market_produces_wider_range(self, simulator: GBMSimulator) -> None:
        """Higher volatility should produce wider prediction ranges."""
        np.random.seed(7)

        # Calm market: tiny price changes
        calm_prices = [78000.0 + np.random.randn() * 5 for _ in range(60)]
        calm_candles = _make_candles(calm_prices)
        calm_result = simulator.predict_next_candle(calm_candles)

        # Volatile market: large price swings
        volatile_prices = [78000.0 + np.random.randn() * 500 for _ in range(60)]
        volatile_candles = _make_candles(volatile_prices)
        volatile_result = simulator.predict_next_candle(volatile_candles)

        assert volatile_result.prediction.width > calm_result.prediction.width

    def test_ewma_fallback_with_small_window(self) -> None:
        """With < 50 returns, GARCH should be skipped and EWMA used instead."""
        simulator = GBMSimulator(
            num_simulations=1_000,
            volatility_lookback=24,
            ewma_span=12,
        )
        np.random.seed(99)
        # 30 candles = 29 returns < 50, will use EWMA fallback
        prices = [78000.0 + np.random.randn() * 50 for _ in range(30)]
        candles = _make_candles(prices)
        result = simulator.predict_next_candle(candles)

        assert result.prediction.lower_bound < result.prediction.upper_bound
        assert result.prediction.width > 0
        assert result.fitting_method == "ewma"

    def test_constant_price_produces_narrow_range(self, simulator: GBMSimulator) -> None:
        """Flat prices with zero volatility should produce a very narrow range."""
        # All prices identical — sigma should be near zero, range near zero
        prices = [78000.0] * 60
        candles = _make_candles(prices)
        result = simulator.predict_next_candle(candles)

        # Range should exist but be extremely narrow
        assert result.prediction.width < 100  # Less than $100 for flat market

    def test_regime_detection_calm_market(self, simulator: GBMSimulator) -> None:
        """A very calm recent period relative to history should detect 'low' regime."""
        np.random.seed(55)
        # First 40 bars: volatile. Last 24 bars: very calm.
        volatile_part = [78000.0 + np.random.randn() * 300 for _ in range(40)]
        calm_part = [volatile_part[-1] + np.random.randn() * 2 for _ in range(24)]
        prices = volatile_part + calm_part
        candles = _make_candles(prices)
        result = simulator.predict_next_candle(candles)

        assert result.volatility_regime == "low"
        assert result.sigma_multiplier <= 1.03

    def test_regime_detection_volatile_market(self, simulator: GBMSimulator) -> None:
        """A very volatile recent period relative to history should detect 'high' regime."""
        np.random.seed(77)
        # First 40 bars: calm. Last 24 bars: very volatile.
        calm_part = [78000.0 + np.random.randn() * 5 for _ in range(40)]
        volatile_part = [calm_part[-1] + np.random.randn() * 500 for _ in range(24)]
        prices = calm_part + volatile_part
        candles = _make_candles(prices)
        result = simulator.predict_next_candle(candles)

        assert result.volatility_regime == "high"
        assert result.sigma_multiplier > 1.03
