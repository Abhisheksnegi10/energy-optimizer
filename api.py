from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os
import json
import numpy as np
import random
from datetime import datetime, timedelta

app = FastAPI(
    title="Smart Energy Optimization API",
    version="1.0",
    description="Predict and optimize energy usage using AI/ML in smart homes."
)

# CORS Setup (allow Vercel Frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # change to "https://your-frontend.vercel.app" in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Load model and features
# ------------------------------------------------------------------
MODEL_PATH = "models/energy_model.pkl"
FEATURES_PATH = "models/features.txt"

model = None
features_list = []

def _ensure_model_loaded():
    global model, features_list
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=500, detail="Model file not found.")
        model = joblib.load(MODEL_PATH)

    if not features_list:
        if not os.path.exists(FEATURES_PATH):
            raise HTTPException(status_code=500, detail="Features file not found.")
        with open(FEATURES_PATH, "r") as f:
            features_list.extend([x.strip() for x in f.readlines() if x.strip()])

# ------------------------------------------------------------------
# Health Check
# ------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "message": "Smart Energy Optimization API is running."}

# ------------------------------------------------------------------
# Predict Endpoint
# ------------------------------------------------------------------
@app.post("/predict")
def predict(request: dict = Body(...)):
    """
    Single-row prediction.
    Input format:
    {
        "features": {
            "Global_reactive_power": 0.12,
            "Voltage": 241.0,
            "Global_intensity": 1.2,
            "Sub_metering_1": 0.0,
            "Sub_metering_2": 1.0,
            "Sub_metering_3": 0.0,
            "hour": 15,
            "day_of_week": 2,
            "month": 7,
            "rolling_mean": 0.5
        },
        "return_metadata": true
    }
    """
    _ensure_model_loaded()

    if "features" not in request:
        raise HTTPException(status_code=400, detail="Missing 'features' in request.")

    input_features = request["features"]

    # Align with model features
    input_df = pd.DataFrame([input_features], columns=features_list)
    y_pred = model.predict(input_df)[0]

    response = {"predicted_global_active_power": float(y_pred)}

    if request.get("return_metadata", False):
        response["used_features"] = input_features
        response["model_features"] = features_list

    return response

# ------------------------------------------------------------------
# Insights Endpoint
# ------------------------------------------------------------------
@app.get("/insights")
def get_insights(horizon: int = 24, shift_fraction: float = 0.8):
    """
    Generate insights (forecast + shift simulation + recommendations).
    Returns pre-generated insights.json if available, else regenerates it.
    """
    insights_path = "insights_output/insights.json"
    if os.path.exists(insights_path):
        with open(insights_path, "r") as f:
            return json.load(f)
    else:
        return {"message": "Insights not yet generated."}

# ------------------------------------------------------------------
# Forecast Endpoint
# ------------------------------------------------------------------
@app.get("/forecast")
def get_forecast(force_regenerate: bool = False, horizon: int = 24, shift_fraction: float = 0.8):
    """
    Return forecast_summary.csv as JSON. 
    If not present or force_regenerate=True, regenerates insights first.
    """
    forecast_path = "insights_output/forecast_summary.csv"

    if not os.path.exists(forecast_path) or force_regenerate:
        if os.path.exists("insights_output/insights.json"):
            os.remove("insights_output/insights.json")
        return {"message": "Forecast data not found. Regenerate insights first."}

    df = pd.read_csv(forecast_path)
    return df.to_dict(orient="records")

# ------------------------------------------------------------------
# Insights File Endpoint
# ------------------------------------------------------------------
@app.get("/insights_file")
def get_insights_file():
    path = "insights_output/insights.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Insights file not found.")
    with open(path, "r") as f:
        return json.load(f)

# ------------------------------------------------------------------
# NEW: Virtual Smart Home Simulation
# ------------------------------------------------------------------
@app.post("/simulate_home")
def simulate_home(profile: dict = Body(...)):
    """
    Simulate a virtual smart home environment based on user input.
    Example input:
    {
        "region": "Delhi",
        "occupants": 3,
        "appliances": ["AC", "Fridge", "TV"],
        "intensity": "High"
    }
    """

    _ensure_model_loaded()

    # base multipliers
    base_voltage = 220 + random.uniform(-10, 10)
    base_intensity = {"Low": 1.0, "Medium": 2.5, "High": 4.0}.get(profile["intensity"], 2.0)

    # simulate one sample
    sample = {
        "Global_reactive_power": round(random.uniform(0.05, 0.25), 3),
        "Voltage": round(base_voltage, 2),
        "Global_intensity": round(base_intensity, 2),
        "Sub_metering_1": random.randint(0, 2),
        "Sub_metering_2": random.randint(0, 2),
        "Sub_metering_3": random.randint(0, 2),
        "hour": random.randint(0, 23),
        "day_of_week": random.randint(0, 6),
        "month": random.randint(1, 12),
        "rolling_mean": round(random.uniform(0.1, 1.5), 3),
    }

    # predict using model
    y_pred = model.predict(pd.DataFrame([sample]))[0]

    # recommendations
    region = profile.get("region", "your area")
    occupants = profile.get("occupants", 2)
    appliances = profile.get("appliances", [])

    appliance_list = ", ".join(appliances) if appliances else "common appliances"

    tips = [
        f"Shift {appliance_list} usage to after 10 PM to reduce peak cost in {region}.",
        f"Set AC to 24°C for optimal efficiency with {occupants} occupants.",
        f"Unplug idle devices — they account for ~5% standby power waste.",
    ]

    return {
        "simulated_input": sample,
        "predicted_global_active_power": float(y_pred),
        "recommendations": tips,
        "timestamp": datetime.now().isoformat()
    }

# ------------------------------------------------------------------
# Root Message
# ------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Welcome to Smart Energy Optimization API!"}
