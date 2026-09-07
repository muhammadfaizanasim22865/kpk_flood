"""
Run this AFTER verify_features.py has already produced verification_results.csv.
No Earth Engine needed — this just analyzes the numbers you already have,
to pinpoint exactly which raw feature is driving the mismatches.

Usage:
    python scripts\\summarize_verification.py
    (run from the project folder, or pass --csv path\\to\\verification_results.csv)
"""

import argparse
import pandas as pd

RAW_FEATURES = [
    "aspect", "curvature", "distance_to_drainage", "distance_to_rivers",
    "elevation", "ndvi", "rainfall_frequency", "slope", "twi",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="verification_results.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    print("Per-feature comparison across all test points:\n")
    print(f"{'feature':25s} {'mean CSV':>14s} {'mean GEE':>14s} {'mean abs diff':>15s} {'mean rel%':>12s} {'max rel%':>10s}")
    print("-" * 95)

    for feat in RAW_FEATURES:
        csv_col = f"csv_{feat}"
        gee_col = f"gee_{feat}"
        rel_col = f"reldiff_{feat}"
        if csv_col not in df.columns:
            continue
        mean_csv = df[csv_col].mean()
        mean_gee = df[gee_col].mean()
        abs_diff = (df[csv_col] - df[gee_col]).abs().mean()
        mean_rel = df[rel_col].mean() * 100
        max_rel = df[rel_col].max() * 100
        flag = "  <-- LIKELY CULPRIT" if max_rel > 100 else ""
        print(f"{feat:25s} {mean_csv:14.4f} {mean_gee:14.4f} {abs_diff:15.4f} {mean_rel:11.1f}% {max_rel:9.1f}%{flag}")

    print("\nRaw side-by-side for first 5 rows, worst feature only:")
    worst_feat = None
    worst_val = -1
    for feat in RAW_FEATURES:
        rel_col = f"reldiff_{feat}"
        if rel_col in df.columns and df[rel_col].max() > worst_val:
            worst_val = df[rel_col].max()
            worst_feat = feat
    if worst_feat:
        print(f"\nWorst feature: {worst_feat}")
        print(df[["latitude", "longitude", f"csv_{worst_feat}", f"gee_{worst_feat}", f"reldiff_{worst_feat}"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
