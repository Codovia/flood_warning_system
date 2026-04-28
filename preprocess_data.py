import pandas as pd
import requests
import numpy as np
from typing import Dict, List, Optional

output_file = 'data/karnataka_flood_ready.csv'

DISTRICT_COORDS: Dict[str, tuple] = {
    'UDUPI': (13.3409, 74.7421),
    'DAKSHINA KANNADA': (12.9141, 74.8560),
    'UTTARA KANNADA': (14.7937, 74.6869),
    'SHIVAMOGGA': (13.9299, 75.5681),
    'BENGALURU URBAN': (12.9716, 77.5946),
}


def assign_risk_level(df: pd.DataFrame) -> pd.Series:
    conditions = [
        (df['Rainfall_mm'] > 100) | ((df['Rainfall_mm'] > 60) & (df['Soil_Moisture'] > 0.35)),
        (df['Rainfall_mm'] > 40) & (df['Rainfall_mm'] <= 100),
        (df['Rainfall_mm'] <= 40),
    ]
    choices = ['High', 'Medium', 'Low']
    return np.select(conditions, choices, default='Low')


def fetch_daily_weather(lat: float, lon: float, retries: int = 3) -> Optional[dict]:
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        "&start_date=2010-01-01&end_date=2024-12-31"
        "&daily=rain_sum,soil_moisture_0_to_7cm_mean&timezone=auto"
    )

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            data = response.json()
            if 'daily' not in data:
                raise ValueError("Missing 'daily' key in API response")
            required_keys = {'time', 'rain_sum', 'soil_moisture_0_to_7cm_mean'}
            if not required_keys.issubset(set(data['daily'].keys())):
                raise ValueError('Missing required daily keys in API response')
            return data
        except Exception as exc:
            print(f"Attempt {attempt}/{retries} failed: {exc}")

    return None

def fetch_modern_flood_data():
    print("1. Fetching modern climate data (2010-2024) via Satellite API...")
    district_frames: List[pd.DataFrame] = []

    for district, (lat, lon) in DISTRICT_COORDS.items():
        print(f"  - Fetching {district}...")
        data = fetch_daily_weather(lat, lon)
        if data is None:
            print(f"    Skipping {district} due to repeated API failures.")
            continue

        df = pd.DataFrame(
            {
                'Date': pd.to_datetime(data['daily']['time']),
                'Rainfall_mm': data['daily']['rain_sum'],
                'Soil_Moisture': data['daily']['soil_moisture_0_to_7cm_mean'],
            }
        ).dropna()

        df['Risk_Level'] = assign_risk_level(df)
        df['DISTRICT'] = district
        df['YEAR'] = df['Date'].dt.year
        district_frames.append(df)

    if not district_frames:
        print("Error: No district data could be downloaded.")
        return

    print("2. Combining district-level datasets...")
    full_df = pd.concat(district_frames, ignore_index=True)
    full_df.sort_values(['DISTRICT', 'Date'], inplace=True)

    full_df.to_csv(output_file, index=False)
    print(f"\nSuccess! Modern dataset (with soil moisture) saved to {output_file}")
    print(full_df.tail())

if __name__ == "__main__":
    fetch_modern_flood_data()