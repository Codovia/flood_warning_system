"""Unit tests for train_model.py model training and evaluation."""

import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def build_and_evaluate_model(df: pd.DataFrame):
    """Replicate the training workflow from train_model.py."""
    X = df[["Rainfall_mm", "Soil_Moisture"]]
    y = df["Risk_Level"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy


def _synthetic_dataset(n: int = 300) -> pd.DataFrame:
    """Return a small deterministic dataset that mirrors real label logic."""
    rng = np.random.default_rng(0)
    rainfall = rng.uniform(0, 200, n)
    moisture = rng.uniform(0, 0.6, n)

    conditions = [
        (rainfall > 100) | ((rainfall > 60) & (moisture > 0.35)),
        (rainfall > 40) & (rainfall <= 100),
        rainfall <= 40,
    ]
    choices = ["High", "Medium", "Low"]
    risk = np.select(conditions, choices, default="Low")

    return pd.DataFrame(
        {"Rainfall_mm": rainfall, "Soil_Moisture": moisture, "Risk_Level": risk}
    )


class TestModelTraining:
    def test_model_returns_classifier(self):
        df = _synthetic_dataset()
        model, _ = build_and_evaluate_model(df)
        assert isinstance(model, RandomForestClassifier)

    def test_model_accuracy_above_threshold(self):
        """Model accuracy on synthetic (noise-free) labels should be high."""
        df = _synthetic_dataset(500)
        _, accuracy = build_and_evaluate_model(df)
        assert accuracy >= 0.90, f"Accuracy too low: {accuracy:.2%}"

    def test_model_predicts_all_three_classes(self):
        df = _synthetic_dataset(500)
        model, _ = build_and_evaluate_model(df)
        sample = pd.DataFrame(
            {
                "Rainfall_mm": [5.0, 70.0, 150.0],
                "Soil_Moisture": [0.1, 0.2, 0.5],
            }
        )
        preds = model.predict(sample)
        assert set(preds) == {"Low", "Medium", "High"}

    def test_model_predict_input_shape(self):
        """predict() on a single-row DataFrame must return one label."""
        df = _synthetic_dataset()
        model, _ = build_and_evaluate_model(df)
        single = pd.DataFrame({"Rainfall_mm": [80.0], "Soil_Moisture": [0.3]})
        result = model.predict(single)
        assert len(result) == 1

    def test_model_serialise_and_reload(self):
        """Saved model must produce identical predictions after reload."""
        df = _synthetic_dataset()
        model, _ = build_and_evaluate_model(df)
        sample = pd.DataFrame({"Rainfall_mm": [120.0], "Soil_Moisture": [0.4]})
        original_pred = model.predict(sample)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            joblib.dump(model, tmp_path)
            reloaded = joblib.load(tmp_path)
            assert reloaded.predict(sample)[0] == original_pred[0]
        finally:
            os.unlink(tmp_path)
