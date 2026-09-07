"""
Loads the pre-trained XGBoost model and produces a susceptibility score.
Nothing here retrains or refits anything.

Uses the NATIVE XGBoost booster format (kpk_flood_susceptibility_xgboost.json),
not the original joblib/sklearn pipeline. This was converted directly from the
original .joblib (verified to produce bit-identical predict_proba output —
max abs diff 0.0 across the full training CSV) specifically to drop the
scikit-learn + scipy dependency, which was pushing the Vercel function bundle
past its 500MB limit (698MB before this change). The original pipeline had
only one step (the XGBClassifier itself, no scaler/preprocessing), so this
loses nothing.
"""

import os
import xgboost as xgb

from .feature_engineering import ENGINEERED_FEATURE_ORDER

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "kpk_flood_susceptibility_xgboost.json")

_booster = None


def get_model():
    global _booster
    if _booster is None:
        _booster = xgb.Booster()
        _booster.load_model(_MODEL_PATH)
    return _booster


def predict_susceptibility(engineered: dict) -> dict:
    """
    engineered: dict with keys = ENGINEERED_FEATURE_ORDER
    returns: {"susceptibility_score": float, "prediction": int, "risk_level": str}
    """
    booster = get_model()
    vector = [[engineered[k] for k in ENGINEERED_FEATURE_ORDER]]
    dmatrix = xgb.DMatrix(vector, feature_names=ENGINEERED_FEATURE_ORDER)

    score = float(booster.predict(dmatrix)[0])  # binary:logistic -> probability directly
    prediction = int(score >= 0.5)

    return {
        "susceptibility_score": round(score, 6),
        "prediction": prediction,
        "risk_level": risk_level_from_score(score),
    }


def risk_level_from_score(score: float) -> str:
    """
    Illustrative thresholds only — NOT independently calibrated or
    validated against local flood-frequency data. The model itself was
    not calibrated (see model_metadata.json: no CalibratedClassifierCV
    step). Treat these bucket boundaries as a starting point for the UI,
    not a certified risk classification. Revisit with domain experts /
    provincial disaster-management authorities before public release.
    """
    if score < 0.30:
        return "Low"
    elif score < 0.55:
        return "Moderate"
    elif score < 0.80:
        return "High"
    else:
        return "Very High"
