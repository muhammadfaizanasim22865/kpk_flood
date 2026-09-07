# KPK Flood Susceptibility — Deployment Guide

## 1. Architecture

```
User clicks map / enters coordinates  (Lovable frontend)
        │  POST /api/predict { latitude, longitude }
        ▼
FastAPI backend (Vercel serverless function, api/index.py)
        │
        ├─ lib/gee_auth.py     → authenticates to Earth Engine (service account)
        ├─ lib/gee_features.py → queries GEE, returns 9 raw features for the point
        ├─ lib/feature_engineering.py → derives the 11 engineered features
        │                                (exact match to training)
        └─ lib/model.py        → loads model/kpk_flood_susceptibility_xgboost.joblib,
                                   returns susceptibility_score + risk_level
        ▼
JSON response → frontend renders score, risk level, and the pin on the map
```

Frontend and backend live in one Vercel project: `api/index.py` is a Python
serverless function, everything else is the static Lovable build. `vercel.json`
routes `/api/*` to the function and everything else to the static site.

## 2. Runtime compatibility verification — READ THIS FIRST

**I could not fully verify this for you, and you should not deploy until you have.**
Here's exactly why, and what I did check.

**What I confirmed (static code review, no GEE needed):**
- The model is an `sklearn.Pipeline` wrapping `XGBClassifier`. Its
  `feature_names_in_` is `['curvature', 'elevation', 'ndvi', 'rainfall_frequency',
  'slope', 'twi', 'aspect_sin', 'aspect_cos', 'is_flat', 'log_distance_to_drainage',
  'log_distance_to_rivers']` — this matches `model_metadata.json` and
  `deployment_config.json` exactly, and I reproduced it end-to-end against the CSV
  (loaded the model, engineered features, called `predict_proba`, cross-checked
  against `target`) — it works and gives sane, high-AUC output.
- The feature engineering formulas in `lib/feature_engineering.py` are copied
  verbatim from `deployment_config.json` and the notebook's Phase 2 cell.
- The 9-band image construction in `lib/gee_features.py` is copied verbatim from
  your supplied `Google Earth Engine runtime feature extraction.py` — same
  datasets, same masks, same scales (FABDEM mosaic → EPSG:3857 30m; MERIT Hydro
  `upa` threshold 0.5 → distance transform at 100m; JRC GSW occurrence >50% →
  distance transform at 500m; Landsat 8 SR NDVI median 2021–2023; CHIRPS >10mm
  day-count 2010–2023). I confirmed this is also the **last** cell in your
  notebook, i.e. the final iteration after several earlier drafts.

**What I found — and could not resolve myself:**
Your notebook actually *attempted* this exact verification twice (search for
`CSV_PATH` in the notebook). **Both attempts crashed before producing a single
comparison number:**
1. First attempt: `EEException: Element.toDictionary: Parameter 'element' is
   required and may not be null` — this means `.sample(...).first()` returned an
   *empty* result for the very first test coordinate. `ee.Image.sample()` drops a
   pixel entirely if *any* band is masked (e.g. cloud-masked NDVI, or a point
   right at the edge of a source dataset's footprint) — it doesn't fail loudly,
   it just silently returns nothing, and `.first()` on nothing throws this error.
2. Second attempt: referenced `ee.Image("FABDEM/HIGH")`, an asset ID from an
   *earlier, different* draft in the notebook that doesn't exist — unrelated bug,
   but it also means this attempt never ran either.

So **there is currently no confirmed evidence that the runtime extraction
pipeline reproduces the values the model was actually trained on.** I want to be
direct about that rather than assume it's fine.

**What I changed, and why:**
In `lib/gee_features.py` I replaced `.sample(...).first()` with
`.reduceRegion(ee.Reducer.first(), ...)`. This uses the *identical* underlying
image (same bands, same masks, same scales — nothing about the feature
computation changes), but returns a value **per band**, some of which may be
`None`, instead of silently dropping the whole point. The backend then raises a
specific, catchable error naming exactly which band was null, instead of an
opaque crash. This fixes the specific bug that crashed the notebook's first
validation attempt, but it does **not** by itself prove the numeric values match
the CSV — only running the extraction and comparing values does that.

**What you need to do before trusting this in production:**
Run `scripts/verify_features.py` yourself, with real Earth Engine credentials
(this sandbox has no network access to `earthengine.googleapis.com` and no
service-account key, so I genuinely cannot run it for you). It:
- Samples N random rows from `kpk_flood_susceptibility_dataset.csv`
- Re-extracts the same coordinates via the runtime pipeline
- Diffs raw features, engineered features, and final model scores
- Flags anything outside a tolerance (default 5%)

See §8 "Testing procedure" for the exact command. **Do not point this at
production traffic until that script reports all points within tolerance.** If
it reports systematic mismatches, the most likely causes, in order of
likelihood, are: (a) NDVI compositing differences if the original CSV used a
different date range/cloud-mask than the final notebook cell, (b) the TWI
formula's `cell_size_m = ee.Image.pixelArea().sqrt()` resolving to a different
effective pixel size depending on which band drives the default output
projection at `reduceRegion` time, or (c) MERIT Hydro `upa` interpretation. If
you hit any of these, I'm happy to help debug once you have real output values.

## 3. Backend

See `api/index.py`. Minimal single endpoint:

`POST /api/predict`
```json
{ "latitude": 34.0151969, "longitude": 71.5251716 }
```
→
```json
{
  "latitude": 34.0151969,
  "longitude": 71.5251716,
  "susceptibility_score": 0.953306,
  "prediction": 1,
  "risk_level": "Very High",
  "features": {
    "aspect": 158.09, "curvature": 3.0, "distance_to_drainage": 1000.0,
    "distance_to_rivers": 38581.08, "elevation": 288, "ndvi": 0.333,
    "rainfall_frequency": 276, "slope": 3.0, "twi": 8.51
  },
  "disclaimer": "This score reflects historical flood susceptibility ..."
}
```

Note the field is called **`susceptibility_score`**, not "probability" — the
model was never run through a calibration step (e.g. `CalibratedClassifierCV`),
so `predict_proba` output shouldn't be presented to end users as a calibrated
probability. The risk-level buckets (`Low`/`Moderate`/`High`/`Very High`) in
`lib/model.py` are placeholder thresholds I picked for a reasonable-looking UI —
they are **not** independently validated and should be reviewed against real
flood outcomes before being treated as authoritative.

Latitude/longitude are validated server-side against KPK's rough bounding box
(31–36.5°N, 69–74.5°E) so obviously out-of-region queries fail fast with a 422
rather than burning an Earth Engine call.

## 4. Earth Engine authentication

Use a Google Cloud **service account**, not personal OAuth:

1. In the Google Cloud project already referenced in your scripts
   (`my-kpk-flood-project-507412`, or your own), create a service account.
2. Register it for Earth Engine access:
   https://signup.earthengine.google.com/#!/service_accounts
3. Create and download a JSON key for it.
4. Base64-encode the key file:
   ```bash
   base64 -i service-account.json | tr -d '\n'
   ```
5. In Vercel → Project Settings → Environment Variables, add:
   - `GEE_SERVICE_ACCOUNT_KEY_B64` = the base64 string (mark as Sensitive)
   - `GEE_PROJECT_ID` = your Earth Engine project id
   - `ALLOWED_ORIGINS` = your deployed frontend URL

These are read server-side only (`lib/gee_auth.py`); nothing touches the
browser. Never commit the JSON key or the base64 string to git — `.env.example`
is a template, not a real secret file.

## 5. Vercel project structure

```
project/
├── api/
│   └── index.py                # FastAPI app — Vercel Python serverless function
├── lib/
│   ├── __init__.py
│   ├── feature_engineering.py  # raw → engineered features, exact training match
│   ├── gee_features.py         # 9-band GEE image + per-point extraction
│   ├── gee_auth.py             # service-account auth
│   └── model.py                # loads joblib, predicts, buckets risk
├── model/
│   ├── kpk_flood_susceptibility_xgboost.joblib
│   ├── model_metadata.json
│   └── deployment_config.json
├── scripts/
│   └── verify_features.py      # run yourself before deploying — see §2
├── requirements.txt
├── vercel.json
└── .env.example
```

Your Lovable-exported frontend code (typically a Vite/React app that builds to
`dist/`) goes at the project root alongside this structure — `vercel.json`
already assumes a `package.json` + `dist/` build output at the root; adjust the
`distDir` in `vercel.json` if Lovable's export differs.

**One honest caveat on hosting**: `scikit-learn` + `xgboost` + `earthengine-api`
together are not tiny. If the packaged function exceeds Vercel's size/cold-start
limits on your plan, the fallback is to run this same FastAPI app on a small
always-on host (Render, Fly.io, a $5 VPS) and only keep the static frontend on
Vercel, pointing `/api/*` at that host instead. The code doesn't change either
way — only where `api/index.py` runs.

## 6. Frontend integration (what to tell Lovable)

The frontend should:
- Show an interactive map centered on KPK (e.g. Leaflet or Mapbox GL via a free tier)
- Let the user click a point, or search/enter lat/lon
- On selection, `POST` to `/api/predict` with `{ latitude, longitude }`
- Show a loading state (the Earth Engine round trip can take a few seconds)
- Render: susceptibility score, risk-level badge, a plain-language explanation,
  and the selected point as a marker on the map
- Show the returned `disclaimer` text near the result, not buried in a footer

## 7. Required wording

Per your requirement, the UI copy must describe this as **historical flood
susceptibility** or **relative susceptibility** — never as a flood *forecast* or
a *prediction that a flood will happen*. The backend already returns a
`disclaimer` field with this exact framing baked in; surface it to the user
rather than writing separate copy that could drift from it.

## 8. Testing procedure (do this before going live)

```bash
cd project
pip install -r requirements.txt earthengine-api pandas

# One-time interactive auth for YOUR account, just to run this check
earthengine authenticate

python scripts/verify_features.py \
  --csv /path/to/kpk_flood_susceptibility_dataset.csv \
  --project my-kpk-flood-project-507412 \
  --n 15
```

This prints a per-point comparison and writes `verification_results.csv` with
CSV vs. GEE values for all 9 raw features plus the two model scores side by
side. Fix any flagged mismatches (see §2 for likely causes) before wiring the
backend to production traffic.

Once that passes, smoke-test the deployed API directly:
```bash
curl -X POST https://your-app.vercel.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{"latitude": 34.0151969, "longitude": 71.5251716}'
```
Compare the `features` block in the response against the corresponding CSV row
for that same coordinate (row 0 in the CSV — this is the test point used in the
supplied runtime script) as a final sanity check before sharing the URL.
