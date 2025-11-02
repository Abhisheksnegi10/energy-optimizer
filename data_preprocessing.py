# ============================================
# ENERGY OPTIMIZATION PROJECT — DATA PIPELINE
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_preprocess(data_path='data/household_power_consumption.txt'):
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv(
        data_path,
        sep=';',
        na_values='?',
        parse_dates={'datetime': ['Date', 'Time']},
        infer_datetime_format=True,
        low_memory=False
    )

    print("Data loaded successfully ✅")

    # Convert to numeric
    num_cols = ['Global_active_power','Global_reactive_power','Voltage',
                'Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop missing rows
    df.dropna(inplace=True)

    # Extract useful time features
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.dayofweek

    # Add rolling mean as feature (short-term energy trend)
    df['rolling_mean'] = df['Global_active_power'].rolling(window=3, min_periods=1).mean()

    # Save preprocessed dataset
    os.makedirs('processed', exist_ok=True)
    df.to_csv('processed/energy_clean.csv', index=False)
    print("Preprocessed data saved to 'processed/energy_clean.csv'")

    return df


if __name__ == "__main__":
    data = load_and_preprocess()
    print(data.head())

    # Basic EDA visualization
    plt.figure(figsize=(10,4))
    sns.lineplot(x='hour', y='Global_active_power', data=data.sample(5000))
    plt.title("Hourly Energy Usage Pattern")
    plt.show()

    plt.figure(figsize=(8,6))
    sns.heatmap(data.corr(), cmap='coolwarm', annot=True)
    plt.title("Feature Correlation Heatmap")
    plt.show()
