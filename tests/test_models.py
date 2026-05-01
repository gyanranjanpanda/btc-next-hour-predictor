"""Tests for domain models: Candle, Prediction, BacktestResult."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.domain.models import BacktestResult, Candle, Prediction


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def sample_candle() -> Candle:
    return Candle(
        timestamp=datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc),
        open_price=78000.0,
        high_price=78500.0,
        low_price=77800.0,
        close_price=78200.0,
        volume=1234.5,
    )


@pytest.fixture
def hit_prediction() -> Prediction:
    """Prediction where actual_close falls inside the range."""
    return Prediction(
        timestamp=datetime(2026, 5, 1, 13, 0, tzinfo=timezone.utc),
        lower_bound=77500.0,
        upper_bound=78500.0,
        actual_close=78000.0,
        confidence_interval=0.95,
    )


@pytest.fixture
def miss_below_prediction() -> Prediction:
    """Prediction where actual_close falls below the range."""
    return Prediction(
        timestamp=datetime(2026, 5, 1, 13, 0, tzinfo=timezone.utc),
        lower_bound=78000.0,
        upper_bound=79000.0,
        actual_close=77500.0,
        confidence_interval=0.95,
    )


@pytest.fixture
def miss_above_prediction() -> Prediction:
    """Prediction where actual_close falls above the range."""
    return Prediction(
        timestamp=datetime(2026, 5, 1, 13, 0, tzinfo=timezone.utc),
        lower_bound=77000.0,
        upper_bound=78000.0,
        actual_close=78500.0,
        confidence_interval=0.95,
    )


@pytest.fixture
def pending_prediction() -> Prediction:
    """Prediction without actual_close (future bar)."""
    return Prediction(
        timestamp=datetime(2026, 5, 1, 13, 0, tzinfo=timezone.utc),
        lower_bound=77500.0,
        upper_bound=78500.0,
        confidence_interval=0.95,
    )


# ── Candle Tests ──────────────────────────────────────────────────────

class TestCandle:
    def test_candle_is_immutable(self, sample_candle: Candle) -> None:
        with pytest.raises(Exception):
            sample_candle.close_price = 99999.0  # type: ignore[misc]

    def test_candle_fields(self, sample_candle: Candle) -> None:
        assert sample_candle.open_price == 78000.0
        assert sample_candle.close_price == 78200.0
        assert sample_candle.volume == 1234.5


# ── Prediction Tests ─────────────────────────────────────────────────

class TestPrediction:
    def test_width_calculation(self, hit_prediction: Prediction) -> None:
        assert hit_prediction.width == pytest.approx(1000.0)

    def test_contains_actual_hit(self, hit_prediction: Prediction) -> None:
        assert hit_prediction.contains_actual is True

    def test_contains_actual_miss_below(self, miss_below_prediction: Prediction) -> None:
        assert miss_below_prediction.contains_actual is False

    def test_contains_actual_miss_above(self, miss_above_prediction: Prediction) -> None:
        assert miss_above_prediction.contains_actual is False

    def test_contains_actual_pending(self, pending_prediction: Prediction) -> None:
        assert pending_prediction.contains_actual is False

    def test_winkler_score_hit_equals_width(self, hit_prediction: Prediction) -> None:
        """When the actual price is inside the range, Winkler = width."""
        assert hit_prediction.winkler_score == pytest.approx(1000.0)

    def test_winkler_score_miss_below_adds_penalty(self, miss_below_prediction: Prediction) -> None:
        """Miss below: penalty = (2/alpha) * (lower - actual)."""
        alpha = 0.05
        width = 1000.0
        distance = 78000.0 - 77500.0  # 500
        expected = width + (2.0 / alpha) * distance
        assert miss_below_prediction.winkler_score == pytest.approx(expected)

    def test_winkler_score_miss_above_adds_penalty(self, miss_above_prediction: Prediction) -> None:
        """Miss above: penalty = (2/alpha) * (actual - upper)."""
        alpha = 0.05
        width = 1000.0
        distance = 78500.0 - 78000.0  # 500
        expected = width + (2.0 / alpha) * distance
        assert miss_above_prediction.winkler_score == pytest.approx(expected)

    def test_winkler_score_pending_returns_none(self, pending_prediction: Prediction) -> None:
        assert pending_prediction.winkler_score is None

    def test_prediction_at_boundary_lower(self) -> None:
        """Actual exactly at lower_bound should count as a hit."""
        pred = Prediction(
            timestamp=datetime(2026, 5, 1, 13, 0, tzinfo=timezone.utc),
            lower_bound=78000.0,
            upper_bound=79000.0,
            actual_close=78000.0,
        )
        assert pred.contains_actual is True
        assert pred.winkler_score == pytest.approx(1000.0)

    def test_prediction_at_boundary_upper(self) -> None:
        """Actual exactly at upper_bound should count as a hit."""
        pred = Prediction(
            timestamp=datetime(2026, 5, 1, 13, 0, tzinfo=timezone.utc),
            lower_bound=78000.0,
            upper_bound=79000.0,
            actual_close=79000.0,
        )
        assert pred.contains_actual is True
        assert pred.winkler_score == pytest.approx(1000.0)


# ── BacktestResult Tests ─────────────────────────────────────────────

class TestBacktestResult:
    def test_backtest_result_fields(self) -> None:
        result = BacktestResult(
            total_predictions=720,
            coverage=0.95,
            average_width=1252.0,
            mean_winkler_score=1721.0,
        )
        assert result.total_predictions == 720
        assert result.coverage == pytest.approx(0.95)
        assert result.average_width == pytest.approx(1252.0)
        assert result.mean_winkler_score == pytest.approx(1721.0)

    def test_backtest_result_is_immutable(self) -> None:
        result = BacktestResult(
            total_predictions=100,
            coverage=0.90,
            average_width=500.0,
            mean_winkler_score=800.0,
        )
        with pytest.raises(Exception):
            result.coverage = 0.99  # type: ignore[misc]
