import pandas as pd


def test_annual_aggregation_from_daily_data():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-01", "2020-02-01", "2021-01-01"]),
            "Rainfall_mm": [10.0, 15.5, 5.0],
        }
    )

    annual_df = df.groupby(df["Date"].dt.year)["Rainfall_mm"].sum().reset_index()
    annual_df.columns = ["YEAR", "ANNUAL"]

    assert annual_df.to_dict(orient="records") == [
        {"YEAR": 2020, "ANNUAL": 25.5},
        {"YEAR": 2021, "ANNUAL": 5.0},
    ]
