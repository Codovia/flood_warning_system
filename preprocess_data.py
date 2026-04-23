import pandas as pd
import requests
import numpy as np

output_file = 'data/karnataka_flood_ready.csv'

def fetch_modern_flood_data():
    print("1. Fetching modern climate data (2010-2024) via Satellite API...")
    # Coordinates for Udupi region
    lat, lon = 13.3409, 74.7421
    
    # We fetch daily rain and soil moisture (key for drainage capacity)
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2010-01-01&end_date=2024-01-01&daily=rain_sum,soil_moisture_0_to_7cm_mean&timezone=auto"
    
    response = requests.get(url)
    if response.status_code != 200:
        print("Error fetching data. Check internet connection.")
        return
        
    data = response.json()
    
    print("2. Processing daily weather into a DataFrame...")
    df = pd.DataFrame({
        'Date': pd.to_datetime(data['daily']['time']),
        'Rainfall_mm': data['daily']['rain_sum'],
        'Soil_Moisture': data['daily']['soil_moisture_0_to_7cm_mean']
    })
    
    # Drop any missing days
    df = df.dropna()
    
    print("3. Engineering realistic Flood Risk levels based on recent climate patterns...")
    # Real-world logic: High rain + saturated soil = Flood. 
    # If soil moisture is high (> 0.3), it takes less rain to cause a flood.
    
    conditions = [
        (df['Rainfall_mm'] > 100) | ((df['Rainfall_mm'] > 60) & (df['Soil_Moisture'] > 0.35)),
        (df['Rainfall_mm'] > 40) & (df['Rainfall_mm'] <= 100),
        (df['Rainfall_mm'] <= 40)
    ]
    choices = ['High', 'Medium', 'Low']
    
    df['Risk_Level'] = np.select(conditions, choices, default='Low')
    df['DISTRICT'] = 'UDUPI' # Tagging it for our system
    df['YEAR'] = df['Date'].dt.year # Needed for Prophet later
    
    # Save the highly accurate, modern dataset
    df.to_csv(output_file, index=False)
    print(f"\nSuccess! Modern dataset (with soil moisture) saved to {output_file}")
    print(df.tail()) # Show the most recent 2024 data

if __name__ == "__main__":
    fetch_modern_flood_data()