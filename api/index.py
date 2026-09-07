"""
FastAPI backend for KPK flood susceptibility.

Single endpoint:
    POST /api/predict   { "latitude": float, "longitude": float }

Flow:
    coordinates -> Earth Engine (9 raw features) -> feature engineering
    -> XGBoost pipeline -> susceptibility score + risk bucket

Deployed on Vercel as a Python serverless function. Vercel's Python
runtime auto-detects an ASGI `app` object exported from api/index.py,
so no extra WSGI/ASGI adapter is needed.
"""

import os
import sys
import time

# Allow `from lib...` imports when this file is executed as the Vercel
# function entrypoint (its cwd/sys.path doesn't include the project root
# by default).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from lib import gee_auth
from lib.gee_features import extract_raw_features, NoDataError
from lib.feature_engineering import engineer_features
from lib.model import predict_susceptibility

app = FastAPI(title="KPK Flood Susceptibility API")

# CORS: restrict this to your actual Lovable/Vercel frontend origin(s)
# in production. "*" is fine for initial testing only.
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_ee_ready = False


def _ensure_ee():
    global _ee_ready
    if not _ee_ready:
        gee_auth.init_earth_engine()
        _ee_ready = True


class PredictRequest(BaseModel):
    latitude: float = Field(..., ge=31.0, le=36.5, description="Must fall within KPK's approximate bounding box")
    longitude: float = Field(..., ge=69.0, le=74.5, description="Must fall within KPK's approximate bounding box")


class PredictResponse(BaseModel):
    latitude: float
    longitude: float
    susceptibility_score: float
    prediction: int
    risk_level: str
    features: dict
    disclaimer: str
    confidence_note: str | None = None


DISCLAIMER = (
    "This score reflects historical flood susceptibility based on terrain, "
    "hydrology, vegetation and long-term rainfall patterns. It is NOT a "
    "real-time flood forecast and does not predict whether a flood will "
    "occur tomorrow or at any specific time."
)

# Known limitation (see README.md, section "Runtime compatibility
# verification"): curvature, aspect, and NDVI extraction have not yet been
# fully reconciled against the original training data. Their combined model
# importance is under ~15%, so most scores are stable, but results in the
# 0.4-0.7 "Moderate/High" range are the least certain and most likely to
# shift as this is refined. Remove this once verify_features.py passes
# cleanly for those features.
CONFIDENCE_NOTE = (
    "This location's score falls in a range where ongoing feature-pipeline "
    "validation could still shift the result. Treat the risk level as "
    "provisional."
)


def _needs_confidence_note(score: float) -> bool:
    return 0.35 <= score <= 0.75


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        _ensure_ee()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Earth Engine auth not configured: {e}")

    try:
        raw_features = extract_raw_features(req.latitude, req.longitude)
    except NoDataError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Earth Engine request failed: {e}")

    try:
        engineered = engineer_features(raw_features)
        result = predict_susceptibility(engineered)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return PredictResponse(
        latitude=req.latitude,
        longitude=req.longitude,
        susceptibility_score=result["susceptibility_score"],
        prediction=result["prediction"],
        risk_level=result["risk_level"],
        features=raw_features,
        disclaimer=DISCLAIMER,
        confidence_note=CONFIDENCE_NOTE if _needs_confidence_note(result["susceptibility_score"]) else None,
    )
