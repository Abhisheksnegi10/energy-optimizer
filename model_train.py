# ==========================================================
# ENERGY OPTIMIZATION PROJECT — MODEL TRAINING MODULE
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================================
# 1️⃣ Load Preprocessed Data
# ==========================================================
def load_data(path='processed/energy_clean.csv'):
    print("Loading preprocessed dataset...")
    df = pd.read_csv(path)
    print(f"Data loaded successfully ✅  Shape: {df.shape}")
    return df


# ==========================================================
# 2️⃣ Feature Selection
# ==========================================================
def select_features(df):
    target = 'Global_active_power'

    features = [
        'Global_reactive_power',
        'Voltage',
        'Global_intensity',
        'Sub_metering_1',
        'Sub_metering_2',
        'Sub_metering_3',
        'hour',
        'day_of_week',
        'month',
        'rolling_mean'
    ]

    X = df[features]
    y = df[target]
    return X, y, features, target


# ==========================================================
# 3️⃣ Train Model
# ==========================================================
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Model trained successfully ✅")

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n📊 Model Performance:")
    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"R²   = {r2:.4f}")

    # Save metrics
    os.makedirs('models', exist_ok=True)
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }
    pd.DataFrame([metrics]).to_csv('models/model_metrics.csv', index=False)

    # Plot predicted vs actual
    plt.figure(figsize=(10,5))
    plt.plot(y_test.values[:300], label='Actual', linewidth=2)
    plt.plot(y_pred[:300], label='Predicted', linestyle='--')
    plt.title("Predicted vs Actual Energy Consumption")
    plt.xlabel("Sample Index")
    plt.ylabel("Energy Usage (kW)")
    plt.legend()
    plt.tight_layout()

    os.makedirs('visuals', exist_ok=True)
    plt.savefig('visuals/predicted_vs_actual.png')
    plt.show()

    # Save model
    joblib.dump(model, 'models/energy_model.pkl')

    # Save feature names for inference
    with open('models/features.txt', 'w') as f:
        for col in X.columns:
            f.write(col + '\n')

    print("\nModel & metadata saved successfully ✅")

    return model


# ==========================================================
# 4️⃣ Run End-to-End
# ==========================================================
if __name__ == "__main__":
    df = load_data()
    X, y, features, target = select_features(df)
    model = train_model(X, y)
