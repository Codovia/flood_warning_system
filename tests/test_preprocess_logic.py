import pandas as pd

from preprocess_data import assign_risk_level


def test_assign_risk_level_low_medium_high():
    df = pd.DataFrame(
        {
            "Rainfall_mm": [20, 70, 120],
            "Soil_Moisture": [0.2, 0.2, 0.3],
        }
    )

    labels = assign_risk_level(df)

    assert labels.tolist() == ["Low", "Medium", "High"]


def test_assign_risk_level_high_on_moist_soil_threshold():
    df = pd.DataFrame(
        {
            "Rainfall_mm": [65],
            "Soil_Moisture": [0.4],
        }
    )

    labels = assign_risk_level(df)

    assert labels.tolist() == ["High"]
