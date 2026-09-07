"""
Run this YOURSELF, locally, with real Earth Engine credentials, before
deploying. It was NOT possible to run this in the environment that
generated this deployment package (no network access to Earth Engine,
no service-account credentials available there).

What it does:
  1. Picks N random rows from kpk_flood_susceptibility_dataset.csv
  2. Re-extracts the 9 raw features for those same coordinates using
     the runtime GEE pipeline (lib/gee_features.py)
  3. Compares raw features, engineered features, and final model
     predictions side by side
  4. Flags anything that differs by more than a tolerance

Why this matters: the training notebook itself contains two attempts
at this exact comparison (cells using CSV_PATH), and BOTH crashed
before producing a single result:
  - one hit "Element.toDictionary: Parameter 'element' is required and
    may not be null" (an empty/masked sample)
  - one referenced a nonexistent asset id, "FABDEM/HIGH", from an
    earlier draft
So there is currently NO confirmed evidence that the runtime
extraction pipeline reproduces the values the model was trained on.
This script is how you get that evidence.

Usage:
    pip install earthengine-api pandas numpy xgboost
    earthengine authenticate      # one-time, interactive, for THIS script only
    python scripts/verify_features.py --csv kpk_flood_susceptibility_dataset.csv --n 15
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ee
import numpy as np
import pandas as pd

from lib.gee_features import extract_raw_features, NoDataError
from lib.feature_engineering import engineer_features, RAW_FEATURE_ORDER, to_model_vector
from lib.model import get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to kpk_flood_susceptibility_dataset.csv")
    parser.add_argument("--n", type=int, default=15, help="Number of random rows to check")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", required=True, help="Your Earth Engine cloud project id")
    parser.add_argument("--rel-tol", type=float, default=0.05, help="Relative tolerance flag threshold (5% default)")
    args = parser.parse_args()

    print("Initializing Earth Engine (interactive auth)...")
    ee.Authenticate()  # opens a browser flow the first time; cached after
    ee.Initialize(project=args.project)

    df = pd.read_csv(args.csv)
    sample = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)

    model = get_model()

    rows = []
    failures = []

    for i, row in sample.iterrows():
        lat, lon = float(row["latitude"]), float(row["longitude"])
        print(f"[{i+1}/{len(sample)}] ({lat:.5f}, {lon:.5f}) ...", end=" ", flush=True)

        try:
            gee_raw = extract_raw_features(lat, lon)
        except NoDataError as e:
            print("NO DATA:", e)
            failures.append({"latitude": lat, "longitude": lon, "error": str(e)})
            continue
        except Exception as e:
            print("ERROR:", e)
            failures.append({"latitude": lat, "longitude": lon, "error": str(e)})
            continue

        csv_raw = {k: float(row[k]) for k in RAW_FEATURE_ORDER}

        rec = {"latitude": lat, "longitude": lon}
        max_rel_diff = 0.0
        for k in RAW_FEATURE_ORDER:
            rec[f"csv_{k}"] = csv_raw[k]
            rec[f"gee_{k}"] = gee_raw[k]
            denom = max(abs(csv_raw[k]), 1e-6)
            rel_diff = abs(csv_raw[k] - gee_raw[k]) / denom
            rec[f"reldiff_{k}"] = round(rel_diff, 4)
            max_rel_diff = max(max_rel_diff, rel_diff)

        # Compare final model predictions too
        csv_vec = [to_model_vector(engineer_features(csv_raw))]
        gee_vec = [to_model_vector(engineer_features(gee_raw))]
        rec["csv_score"] = round(float(model.predict_proba(csv_vec)[0][1]), 4)
        rec["gee_score"] = round(float(model.predict_proba(gee_vec)[0][1]), 4)
        rec["max_rel_diff"] = round(max_rel_diff, 4)
        rec["flag"] = "MISMATCH" if max_rel_diff > args.rel_tol else "ok"

        print(rec["flag"], f"(max rel diff {rec['max_rel_diff']:.1%}, score {rec['csv_score']} vs {rec['gee_score']})")
        rows.append(rec)

    result_df = pd.DataFrame(rows)
    out_path = "verification_results.csv"
    result_df.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print(f"Checked {len(sample)} points: {len(rows)} succeeded, {len(failures)} failed to extract.")
    if len(rows):
        n_mismatch = (result_df["flag"] == "MISMATCH").sum()
        print(f"{n_mismatch}/{len(rows)} points exceeded {args.rel_tol:.0%} relative difference on at least one feature.")
    if failures:
        print("\nFailed points (Earth Engine returned no usable data):")
        for f in failures:
            print(" -", f["latitude"], f["longitude"], "->", f["error"])
    print(f"\nFull results written to {out_path}")
    print("=" * 60)

    if len(rows) == 0:
        print("\nNo points could be verified. Do not deploy until this is resolved.")
    elif failures or (len(rows) and (result_df["flag"] == "MISMATCH").any()):
        print("\nSome points failed or mismatched. Investigate before deploying to production.")
    else:
        print("\nAll checked points matched within tolerance.")


if __name__ == "__main__":
    main()
