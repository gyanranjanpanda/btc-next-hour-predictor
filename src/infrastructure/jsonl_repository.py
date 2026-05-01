from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from ..application.interfaces import IPredictionRepository
from ..domain.models import Prediction


class JsonlPredictionRepository(IPredictionRepository):
    """Append-only JSONL persistence for prediction history."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def save(self, prediction: Prediction) -> None:
        """Appends a single prediction as one JSON line."""
        record = prediction.model_dump()
        record["timestamp"] = record["timestamp"].isoformat()
        with open(self.file_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

    def get_all(self) -> list[Prediction]:
        """Reads every stored prediction, skipping corrupt lines."""
        if not os.path.exists(self.file_path):
            return []

        predictions: list[Prediction] = []
        with open(self.file_path) as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                    record["timestamp"] = datetime.fromisoformat(record["timestamp"])
                    predictions.append(Prediction(**record))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        return predictions
