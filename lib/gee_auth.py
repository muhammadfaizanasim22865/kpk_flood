"""
Google Earth Engine authentication for a headless backend.

Do NOT use the interactive `earthengine authenticate` flow in
production — that stores a personal OAuth token on disk and can't run
on Vercel. Use a Earth Engine-enabled Google Cloud service account
instead, with its private key supplied as a Vercel environment
variable (never committed, never sent to the frontend).

Setup (one-time):
  1. In Google Cloud Console, for the project already used in the
     scripts ("my-kpk-flood-project-507412" or your own), create a
     service account, e.g. kpk-flood-backend@<project>.iam.gserviceaccount.com
  2. Register that service account for Earth Engine access at
     https://signup.earthengine.google.com/#!/service_accounts
  3. Create a JSON key for the service account and download it.
  4. Base64-encode the JSON key file and store it as a Vercel
     environment variable, e.g.:
         base64 -i service-account.json | tr -d '\n'
     Set this as GEE_SERVICE_ACCOUNT_KEY_B64 in the Vercel project
     settings (Production + Preview), marked as a "Secret".
  5. Set GEE_PROJECT_ID to your Earth Engine cloud project id.
"""

import base64
import json
import os

import ee


def init_earth_engine():
    """Call once per cold start, before any ee.* calls."""
    key_b64 = os.environ.get("GEE_SERVICE_ACCOUNT_KEY_B64")
    project_id = os.environ.get("GEE_PROJECT_ID")

    if not key_b64 or not project_id:
        raise RuntimeError(
            "Missing GEE_SERVICE_ACCOUNT_KEY_B64 or GEE_PROJECT_ID environment "
            "variables. These must be set in Vercel project settings, not in code."
        )

    key_json = base64.b64decode(key_b64).decode("utf-8")
    key_dict = json.loads(key_json)

    credentials = ee.ServiceAccountCredentials(
        key_dict["client_email"], key_data=key_json
    )
    ee.Initialize(credentials, project=project_id)
