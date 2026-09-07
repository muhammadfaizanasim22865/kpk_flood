"""
Feature engineering — must exactly mirror the transformations used
during training (see model/model_metadata.json / deployment_config.json
and the training notebook, Phase 2).

Raw features in  -> engineered features out, in the EXACT column order
the model was fit on. Order matters for XGBoost's sklearn wrapper.
"""

import math

# Exact order the model expects (from model.named_steps['model'].feature_names_in_)
ENGINEERED_FEATURE_ORDER = [
    "curvature",
    "elevation",
    "ndvi",
    "rainfall_frequency",
    "slope",
    "twi",
    "aspect_sin",
    "aspect_cos",
    "is_flat",
    "log_distance_to_drainage",
    "log_distance_to_rivers",
]

RAW_FEATURE_ORDER = [
    "aspect",
    "curvature",
    "distance_to_drainage",
    "distance_to_rivers",
    "elevation",
    "ndvi",
    "rainfall_frequency",
    "slope",
    "twi",
]


def engineer_features(raw: dict) -> dict:
    """
    raw: dict with keys = RAW_FEATURE_ORDER (values from GEE or CSV)
    returns: dict with keys = ENGINEERED_FEATURE_ORDER, ready for the model
    """
    missing = [k for k in RAW_FEATURE_ORDER if k not in raw or raw[k] is None]
    if missing:
        raise ValueError(
            f"Cannot engineer features — missing/null raw values: {missing}. "
            f"This usually means Earth Engine returned no data for this pixel "
            f"(e.g. outside dataset coverage, or fully cloud-masked NDVI)."
        )

    aspect = float(raw["aspect"])
    slope = float(raw["slope"])
    distance_to_drainage = float(raw["distance_to_drainage"])
    distance_to_rivers = float(raw["distance_to_rivers"])

    engineered = {
        "curvature": float(raw["curvature"]),
        "elevation": float(raw["elevation"]),
        "ndvi": float(raw["ndvi"]),
        "rainfall_frequency": float(raw["rainfall_frequency"]),
        "slope": slope,
        "twi": float(raw["twi"]),
        "aspect_sin": math.sin(math.radians(aspect)),
        "aspect_cos": math.cos(math.radians(aspect)),
        "is_flat": 1 if slope <= 0.1 else 0,
        "log_distance_to_drainage": math.log1p(distance_to_drainage),
        "log_distance_to_rivers": math.log1p(distance_to_rivers),
    }
    return engineered


def to_model_vector(engineered: dict) -> list:
    """Order the engineered dict into the exact vector the model expects."""
    return [engineered[k] for k in ENGINEERED_FEATURE_ORDER]
