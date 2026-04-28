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

    print("Preparing data for Facebook Prophet...")
    # 2. Prophet REQUIRES exact column names: 'ds' (datestamp) and 'y' (target value)
    # Aggregate daily Rainfall_mm into annual totals per year.
    annual_df = district_df.groupby('YEAR', as_index=False)['Rainfall_mm'].sum()
    annual_df.rename(columns={'YEAR': 'ds', 'Rainfall_mm': 'y'}, inplace=True)

    # Convert the raw year into a standard datetime format (e.g., 2010 becomes 2010-01-01)
    annual_df['ds'] = pd.to_datetime(annual_df['ds'], format='%Y')
    prophet_df = annual_df

    print("Training the Prophet model...")
    # 3. Initialize and train the model
    model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
    model.fit(prophet_df)

    print("Forecasting the next 5 years...")
    # 4. Generate future dates and predict
    # periods=5 means 5 steps into the future. freq='YE' stands for Year End.
    future = model.make_future_dataframe(periods=5, freq='YE')
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

