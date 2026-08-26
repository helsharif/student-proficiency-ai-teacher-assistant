"""Load and serve the persisted XGBoost interaction-success model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "xgboost"


class XGBoostModelService:
    """Thin deployment wrapper around notebook 05's native XGBoost artifact."""

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR) -> None:
        self.model_path = model_dir / "xgboost_first_attempt.json"
        self.metadata_path = model_dir / "xgboost_first_attempt_metadata.json"
        if not self.model_path.is_file() or not self.metadata_path.is_file():
            raise FileNotFoundError(f"Expected XGBoost deployment bundle in {model_dir}")
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.feature_names = [str(name) for name in self.metadata["feature_names"]]
        self.threshold = float(self.metadata["selected_threshold"])
        self.model = XGBClassifier()
        self.model.load_model(self.model_path)

    def predict_probability(self, frame: pd.DataFrame) -> np.ndarray:
        missing = sorted(set(self.feature_names) - set(frame.columns))
        if missing:
            raise ValueError(f"Prediction frame is missing model features: {missing[:5]}")
        probability = self.model.predict_proba(frame[self.feature_names])[:, 1]
        return np.asarray(probability, dtype="float64")
