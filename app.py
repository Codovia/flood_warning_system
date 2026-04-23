import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster
import joblib
import json
import requests
from datetime import datetime

# --- 1. Load Assets ---
@st.cache_resource
def load_model():
    return joblib.load('models/random_forest_flood_model.pkl')

@st.cache_data
def load_geojson():
    with open('data/karnataka_districts.geojson', 'r') as f:
        return json.load(f)

@st.cache_data
def load_historical_data():
    return pd.read_csv('data/karnataka_flood_ready.csv')

model = load_model()
geojson_data = load_geojson()
historical_df = load_historical_data()

# --- 2. Live Data Integration ---
def fetch_live_rainfall():
    # Coordinates for Udupi
    lat, lon = 13.3409, 74.7421
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=rain_sum&timezone=auto"
    try:
        response = requests.get(url)
        data = response.json()
        # Get today's rainfall in mm
        todays_rain = data['daily']['rain_sum'][0] 
        return todays_rain
    except Exception as e:
        st.error("Failed to fetch live data. Please check your internet connection.")
        return 0.0

# --- 3. Historical Context ---
def get_historical_context(current_month_index):
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    month_col = months[current_month_index - 1]
    
    # Filter for Udupi region
    udupi_history = historical_df[historical_df['DISTRICT'] == 'UDUPI'].copy()
    
    if udupi_history.empty:
        return None, month_col
    
    # Extract month from the Date column and filter for the current month
    udupi_history['Date'] = pd.to_datetime(udupi_history['Date'], errors='coerce')
    udupi_history = udupi_history[udupi_history['Date'].dt.month == current_month_index]
    
    if udupi_history.empty:
        return None, month_col
    
    # Aggregate total rainfall per year for this month
    monthly_totals = udupi_history.groupby('YEAR')['Rainfall_mm'].sum().reset_index()
    monthly_totals.columns = ['YEAR', month_col]
    
    # Find the top 3 wettest years for the current month
    top_years = monthly_totals.sort_values(by=month_col, ascending=False).head(3)
    return top_years, month_col

# --- 4. Dashboard UI Setup ---
st.set_page_config(page_title="Live AI Flood Warning System", layout="wide")
st.title("🌊 Live AI-Based Flood Risk & Early Warning System")

# Get current date info
today = datetime.now()
st.markdown(f"**Live Dashboard Status:** Active | **Date:** {today.strftime('%B %d, %Y')}")

# --- Sidebar Controls ---
st.sidebar.header("Data Feed")
data_source = st.sidebar.radio("Select Data Source:", ["Live Satellite Data", "Manual Input"])

rainfall_to_predict = 0.0

if data_source == "Live Satellite Data":
    if st.sidebar.button("Fetch Real-Time Data"):
        with st.spinner("Connecting to weather satellites..."):
            rainfall_to_predict = fetch_live_rainfall()
            st.sidebar.success(f"Live Data Retrieved! Today's Rain: {rainfall_to_predict} mm")
            st.session_state['display_rain'] = rainfall_to_predict
else:
    manual_input = st.sidebar.number_input("Forecasted Daily Rainfall (mm)", min_value=0.0, max_value=500.0, value=150.0, step=10.0)
    st.session_state['display_rain'] = manual_input

soil_moisture_input = st.sidebar.slider("Soil Moisture (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
st.session_state['soil_moisture'] = soil_moisture_input

# --- Main Dashboard Logic ---
if 'display_rain' in st.session_state:
    display_rain = st.session_state['display_rain']
    soil_moisture_val = st.session_state['soil_moisture']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # --- ML Prediction ---
        input_df = pd.DataFrame({'Rainfall_mm': [display_rain], 'Soil_Moisture': [soil_moisture_val]})
        prediction = model.predict(input_df)[0]
        
        st.subheader("🚨 Risk Assessment")
        
        if prediction == "High":
            st.error(f"**HIGH RISK DETECTED**")
            map_color = "red"
        elif prediction == "Medium":
            st.warning(f"**MEDIUM RISK DETECTED**")
            map_color = "orange"
        else:
            st.success(f"**LOW RISK**")
            map_color = "green"

        # --- Geospatial Mapping ---
        # Center map to overview all of Karnataka
        m = folium.Map(location=[14.5, 75.8], zoom_start=6)
        
        # Keep GeoJson layer but make it highly transparent so it's a background element
        folium.GeoJson(
            geojson_data,
            style_function=lambda feature: {
                'fillColor': map_color,
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.1
            }
        ).add_to(m)

        # Hotspot logic (Statewide CSV Pins + Heatmap) for High risk
        if prediction == "High":
            try:
                hotspots_df = pd.read_csv('data/karnataka_flood_hotspots.csv')
                
                # Add HeatMap layer statewide
                heat_data = hotspots_df[['Latitude', 'Longitude']].values.tolist()
                heat_data = [[lat, lon, 1.0] for lat, lon in heat_data]
                HeatMap(heat_data, radius=20, blur=15, gradient={0.4: 'yellow', 0.65: 'orange', 1: 'red'}).add_to(m)

                # Add MarkerCluster for distinct clickable pins covering the whole state
                marker_cluster = MarkerCluster().add_to(m)

                for index, row in hotspots_df.iterrows():
                    popup_html = f"<div style='width: 15rem;'><b>🚨 HIGH DANGER</b><br>{row['Location_Name']}<br><i>District: {row['District']}</i><br><i>Risk Type: {row['Risk_Type'].replace('_', ' ')}</i><br><i>Evacuation routes active</i></div>"
                    folium.Marker(
                        location=[row['Latitude'], row['Longitude']],
                        popup=folium.Popup(popup_html, max_width=300),
                        tooltip=f"Danger Zone: {row['Location_Name']}",
                        icon=folium.Icon(color="red", icon="info-sign")
                    ).add_to(marker_cluster)
            except Exception as e:
                st.error(f"Error loading hotspot data: {e}")
                
        st_folium(m, width=600, height=400)

    with col2:
        # --- Historical Comparison ---
        st.subheader("📖 Historical Context")
        st.info(f"Looking back at the month of **{today.strftime('%B')}**...")
        
        top_historical, month_name = get_historical_context(today.month)
        
        if top_historical is not None:
            st.markdown(f"**Worst flooding years for {month_name}:**")
            for index, row in top_historical.iterrows():
                st.markdown(f"- **{int(row['YEAR'])}:** {row[month_name]} mm of rain")
        else:
            st.write("No historical data available for this region.")
            
        st.markdown("---")
        st.markdown("**How this compares:**")
        st.write("By tracking real-time API feeds against historical maximums, the AI determines if current weather events are anomalous and trigger risk thresholds.")

else:
    st.info("👈 Select a data source in the sidebar to begin.")