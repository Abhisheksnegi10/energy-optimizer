# api.py
"""
FastAPI wrapper for Smart Energy Optimization backend.

Endpoints:
- GET  /insights      -> generate insights + forecast (calls insights_generator.generate_insights)
- GET  /forecast      -> return last forecast_summary.csv (or regenerate insights if missing)
- POST /predict       -> single-row prediction using models/energy_model.pkl

Run locally:
    uvicorn api:app --reload --port 8000
"""

import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
from fastapi.middleware.cors import CORSMiddleware

import joblib
import pandas as pd

# Import helper from your Phase 3 script
from insights_generator import generate_insights, load_model_and_meta

# Config / paths (keep in sync with other scripts)
INSIGHTS_DIR = 'insights_output'
FORECAST_CSV = os.path.join(INSIGHTS_DIR, 'forecast_summary.csv')
MODEL_PATH = 'models/energy_model.pkl'
FEATURES_PATH = 'models/features.txt'
PROCESSED_PATH = 'processed/energy_clean.csv'

app = FastAPI(title="Smart Energy Optimization API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for demo; restrict later to your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for /predict payload
class PredictRequest(BaseModel):
    # dynamic mapping: accept any numeric features, but we'll validate against features.txt at runtime
    features: Dict[str, float]
    # optional: allow a flag to return model metadata
    return_metadata: Optional[bool] = False


# Utility: load model & feature list (cached)
_MODEL = None
_FEATURES = None
def _ensure_model_loaded():
    global _MODEL, _FEATURES
    if _MODEL is None or _FEATURES is None:
        _MODEL, _FEATURES = load_model_and_meta(MODEL_PATH, FEATURES_PATH)
    return _MODEL, _FEATURES

# ----- Endpoints ----- #

@app.get("/health")
async def health():
    return {"status": "ok", "message": "Smart Energy Optimization API is running."}


@app.get("/insights")
async def get_insights(horizon: int = 24, shift_fraction: float = 0.8):
    """
    Generate insights (forecast + shift simulation + recommendations).
    This calls insights_generator.generate_insights which saves outputs into insights_output/.
    Returns the generated insights JSON.
    """
    try:
        insights = generate_insights(horizon=horizon, shift_fraction=shift_fraction)
        return JSONResponse(content=insights)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast")
async def get_forecast(force_regenerate: bool = False, horizon: int = 24, shift_fraction: float = 0.8):
    """
    Return forecast_summary.csv as JSON. If not present or force_regenerate=True, regenerate insights first.
    """
    # Regenerate if forced or missing
    if force_regenerate or not os.path.exists(FORECAST_CSV):
        try:
            generate_insights(horizon=horizon, shift_fraction=shift_fraction)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate insights: {e}")

    # Load CSV and return as JSON
    try:
        df = pd.read_csv(FORECAST_CSV)
        return JSONResponse(content={"forecast": df.to_dict(orient="records")})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read forecast CSV: {e}")


@app.post("/predict")
async def predict(payload: PredictRequest):
    """
    Single-row prediction. Expects payload.features to contain exactly the features in models/features.txt
    Example request body:
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
    try:
        model, features = _ensure_model_loaded()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load error: {e}")

    # Validate input features
    input_features = payload.features
    missing = [f for f in features if f not in input_features]
    extra = [k for k in input_features.keys() if k not in features]

    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
    # We allow extra but will ignore them (with a warning)
    if extra:
        # don't raise; just ignore extra keys
        pass

    # Build dataframe in correct column order
    X = pd.DataFrame([{f: float(input_features.get(f, 0.0)) for f in features}])

    try:
        pred = model.predict(X)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")

    response = {"predicted_global_active_power": float(pred)}
    if payload.return_metadata:
        response["model_path"] = MODEL_PATH
        response["features_used"] = features

    return JSONResponse(content=response)


# Optional: endpoint to download the latest insights.json
@app.get("/insights_file")
async def insights_file():
    json_path = os.path.join(INSIGHTS_DIR, 'insights.json')
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="insights.json not found; run /insights to generate.")
    return FileResponse(json_path, media_type='application/json', filename='insights.json')
