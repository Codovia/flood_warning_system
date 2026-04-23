import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

os.makedirs('models', exist_ok=True)

def train_modern_model():
    print("1. Loading modern satellite data...")
    try:
        df = pd.read_csv('data/karnataka_flood_ready.csv')
    except FileNotFoundError:
        print("Error: data/karnataka_flood_ready.csv not found.")
        return
    
    # 2. Define Features (X) and Target (y)
    # We now use TWO features for much higher real-world accuracy
    X = df[['Rainfall_mm', 'Soil_Moisture']] 
    y = df['Risk_Level']
    
    print("2. Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("3. Training the enhanced Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("4. Evaluating model performance...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n--- Evaluation Metrics ---")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # 5. Save the upgraded model
    joblib.dump(model, 'models/random_forest_flood_model.pkl')
    print("\nSuccess! Modern model saved to models/random_forest_flood_model.pkl")

if __name__ == "__main__":
    train_modern_model()