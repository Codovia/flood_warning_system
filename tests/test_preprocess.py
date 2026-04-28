"""Unit tests for preprocess_data.py risk-level engineering logic."""

import numpy as np
import pandas as pd
import pytest


def compute_risk_level(rainfall_mm: pd.Series, soil_moisture: pd.Series) -> pd.Series:
    """Replicate the threshold logic from preprocess_data.py."""
    conditions = [
        (rainfall_mm > 100) | ((rainfall_mm > 60) & (soil_moisture > 0.35)),
        (rainfall_mm > 40) & (rainfall_mm <= 100),
        (rainfall_mm <= 40),
    ]
    choices = ["High", "Medium", "Low"]
    return pd.Series(np.select(conditions, choices, default="Low"))


class TestRiskLevelEngineering:
    def _make_df(self, rainfall, moisture):
        return pd.DataFrame(
            {"Rainfall_mm": [rainfall], "Soil_Moisture": [moisture]}
        )

    # --- High risk cases ---

    def test_very_heavy_rain_gives_high_risk(self):
        df = self._make_df(150, 0.2)
        result = compute_risk_level(df["Rainfall_mm"], df["Soil_Moisture"])
        assert result.iloc[0] == "High"

    def test_exactly_100mm_not_high(self):
        """Boundary: > 100 means strictly greater than, so 100 should be Medium."""
        df = self._make_df(100, 0.1)
        result = compute_risk_level(df["Rainfall_mm"], df["Soil_Moisture"])
        assert result.iloc[0] == "Medium"

    def test_saturated_soil_with_moderate_rain_is_high(self):
        """61 mm rain + soil moisture > 0.35 → High."""
        df = self._make_df(61, 0.4)
        result = compute_risk_level(df["Rainfall_mm"], df["Soil_Moisture"])
        assert result.iloc[0] == "High"

    def test_moderate_rain_dry_soil_not_high(self):
        """61 mm rain + soil moisture ≤ 0.35 → Medium (not High)."""
        df = self._make_df(61, 0.3)
        result = compute_risk_level(df["Rainfall_mm"], df["Soil_Moisture"])
        assert result.iloc[0] == "Medium"

    # --- Medium risk cases ---

    def test_medium_rain_gives_medium_risk(self):
        df = self._make_df(70, 0.2)
        result = compute_risk_level(df["Rainfall_mm"], df["Soil_Moisture"])
        assert result.iloc[0] == "Medium"

    def test_exactly_41mm_is_medium(self):
        df = self._make_df(41, 0.1)
        result = compute_risk_level(df["Rainfall_mm"], df["Soil_Moisture"])
        assert result.iloc[0] == "Medium"

    # --- Low risk cases ---

    def test_low_rain_gives_low_risk(self):
        df = self._make_df(10, 0.1)
        result = compute_risk_level(df["Rainfall_mm"], df["Soil_Moisture"])
        assert result.iloc[0] == "Low"

    def test_exactly_40mm_is_low(self):
        """Boundary: ≤ 40 mm → Low."""
        df = self._make_df(40, 0.5)
        result = compute_risk_level(df["Rainfall_mm"], df["Soil_Moisture"])
        assert result.iloc[0] == "Low"

    def test_zero_rain_is_low(self):
        df = self._make_df(0, 0.0)
        result = compute_risk_level(df["Rainfall_mm"], df["Soil_Moisture"])
        assert result.iloc[0] == "Low"

    # --- Batch / vectorised ---

    def test_batch_risk_levels(self):
        df = pd.DataFrame(
            {
                "Rainfall_mm": [5, 50, 120, 65],
                "Soil_Moisture": [0.1, 0.2, 0.4, 0.4],
            }
        )
        expected = ["Low", "Medium", "High", "High"]
        result = compute_risk_level(df["Rainfall_mm"], df["Soil_Moisture"])
        assert result.tolist() == expected
