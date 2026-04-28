import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

# Create a folder for our model outputs and plots
os.makedirs('output', exist_ok=True)

def forecast_district_rainfall(district_name="UDUPI"):
    print(f"Loading preprocessed data for {district_name}...")
    try:
        df = pd.read_csv('data/karnataka_flood_ready.csv')
    except FileNotFoundError:
        print("Error: data/karnataka_flood_ready.csv not found. Did you run the Phase 1 script?")
        return
    
    # 1. Filter for the specific district
    # Ensure the district name is uppercase to match the CSV format
    district_df = df[df['DISTRICT'].str.upper() == district_name.upper()].copy()
    
    if district_df.empty:
        print(f"No data found for {district_name}. Check spelling or dataset.")
        return

    print("Preparing annual aggregates for Prophet...")
    if 'Date' not in district_df.columns:
        print("Error: Date column missing in preprocessed dataset.")
        return

    district_df['Date'] = pd.to_datetime(district_df['Date'], errors='coerce')
    district_df = district_df.dropna(subset=['Date'])
    annual_df = district_df.groupby(district_df['Date'].dt.year)['Rainfall_mm'].sum().reset_index()
    annual_df.columns = ['YEAR', 'ANNUAL']

    if annual_df['YEAR'].nunique() < 3:
        print("Error: Need at least 3 years of annual rainfall to run forecasting.")
        return

    prophet_df = annual_df.rename(columns={'YEAR': 'ds', 'ANNUAL': 'y'})
    
    # Convert the raw year into a standard datetime format (e.g., 2010 becomes 2010-01-01)
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], format='%Y')

    print("Training the Prophet model...")
    # 3. Initialize and train the model
    model = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=False)
    model.fit(prophet_df)

    print("Forecasting the next 5 years...")
    # 4. Generate future dates and predict
    # periods=5 means 5 years into the future.
    future = model.make_future_dataframe(periods=5, freq='Y')
    forecast = model.predict(future)

    # 5. Save the numerical forecast data
    forecast_subset = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)
    forecast_subset.to_csv(f'output/{district_name}_forecast.csv', index=False)
    print(f"Numerical forecast saved to output/{district_name}_forecast.csv")

    # 6. Generate and save a visual plot of the forecast
    fig1 = model.plot(forecast)
    plt.title(f"Annual Rainfall Forecast for {district_name}")
    plt.xlabel("Year")
    plt.ylabel("Rainfall (mm)")
    plt.savefig(f'output/{district_name}_forecast_plot.png')
    print(f"Plot saved successfully to output/{district_name}_forecast_plot.png")

if __name__ == "__main__":
    forecast_district_rainfall("UDUPI")

