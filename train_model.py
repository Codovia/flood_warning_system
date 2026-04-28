import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
import joblib
import os
import json

os.makedirs('models', exist_ok=True)

def train_modern_model():
    print("1. Loading modern satellite data...")
    try:
        df = pd.read_csv('data/karnataka_flood_ready.csv')
    except FileNotFoundError:
        print("Error: data/karnataka_flood_ready.csv not found.")
        return
    
    required_columns = {'Date', 'Rainfall_mm', 'Soil_Moisture', 'Risk_Level'}
    if not required_columns.issubset(set(df.columns)):
        print(f"Error: dataset missing required columns: {required_columns - set(df.columns)}")
        return

    print("2. Preparing time-aware training split...")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    if len(df) < 200:
        print("Error: dataset is too small for reliable train/test split.")
        return

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[['Rainfall_mm', 'Soil_Moisture']]
    y_train = train_df['Risk_Level']
    X_test = test_df[['Rainfall_mm', 'Soil_Moisture']]
    y_test = test_df['Risk_Level']
    
    print("3. Training the enhanced Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    print("4. Evaluating model performance...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"\n--- Evaluation Metrics ---")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(f"Balanced Accuracy: {balanced_accuracy * 100:.2f}%")
    print("Classification Report:")
    print(report)
    
    # 5. Save the upgraded model
    joblib.dump(model, 'models/random_forest_flood_model.pkl')
    metrics = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'train_start': str(train_df['Date'].min().date()),
        'train_end': str(train_df['Date'].max().date()),
        'test_start': str(test_df['Date'].min().date()),
        'test_end': str(test_df['Date'].max().date()),
    }
    with open('models/model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("\nSuccess! Modern model saved to models/random_forest_flood_model.pkl")
    print("Metrics saved to models/model_metrics.json")

if __name__ == "__main__":
    train_modern_model()