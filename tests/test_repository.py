"""Tests for the JSONL prediction repository."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone

import pytest

from src.domain.models import Prediction
from src.infrastructure.jsonl_repository import JsonlPredictionRepository


@pytest.fixture
def tmp_jsonl(tmp_path):
    return str(tmp_path / "test_predictions.jsonl")


@pytest.fixture
def sample_prediction() -> Prediction:
    return Prediction(
        timestamp=datetime(2026, 5, 1, 13, 0, tzinfo=timezone.utc),
        lower_bound=77500.0,
        upper_bound=78500.0,
        actual_close=78000.0,
        confidence_interval=0.95,
    )


class TestJsonlPredictionRepository:
    def test_save_creates_file(self, tmp_jsonl: str, sample_prediction: Prediction) -> None:
        repo = JsonlPredictionRepository(tmp_jsonl)
        repo.save(sample_prediction)
        assert os.path.exists(tmp_jsonl)

    def test_save_appends_json_line(self, tmp_jsonl: str, sample_prediction: Prediction) -> None:
        repo = JsonlPredictionRepository(tmp_jsonl)
        repo.save(sample_prediction)
        repo.save(sample_prediction)

        with open(tmp_jsonl) as fh:
            lines = [l for l in fh.readlines() if l.strip()]
        assert len(lines) == 2
        # Each line should be valid JSON
        for line in lines:
            parsed = json.loads(line)
            assert "lower_bound" in parsed
            assert "upper_bound" in parsed

    def test_get_all_returns_saved_predictions(self, tmp_jsonl: str, sample_prediction: Prediction) -> None:
        repo = JsonlPredictionRepository(tmp_jsonl)
        repo.save(sample_prediction)

        loaded = repo.get_all()
        assert len(loaded) == 1
        assert loaded[0].lower_bound == pytest.approx(77500.0)
        assert loaded[0].upper_bound == pytest.approx(78500.0)
        assert loaded[0].actual_close == pytest.approx(78000.0)

    def test_get_all_on_missing_file_returns_empty(self, tmp_jsonl: str) -> None:
        repo = JsonlPredictionRepository(tmp_jsonl)
        assert repo.get_all() == []

    def test_get_all_skips_corrupt_lines(self, tmp_jsonl: str, sample_prediction: Prediction) -> None:
        repo = JsonlPredictionRepository(tmp_jsonl)
        repo.save(sample_prediction)

        # Append a corrupt line
        with open(tmp_jsonl, "a") as fh:
            fh.write("THIS IS NOT JSON\n")

        repo.save(sample_prediction)

        loaded = repo.get_all()
        assert len(loaded) == 2  # Skipped the corrupt line

    def test_roundtrip_preserves_none_actual(self, tmp_jsonl: str) -> None:
        pred = Prediction(
            timestamp=datetime(2026, 5, 1, 14, 0, tzinfo=timezone.utc),
            lower_bound=77000.0,
            upper_bound=79000.0,
            confidence_interval=0.95,
        )
        repo = JsonlPredictionRepository(tmp_jsonl)
        repo.save(pred)

        loaded = repo.get_all()
        assert len(loaded) == 1
        assert loaded[0].actual_close is None
