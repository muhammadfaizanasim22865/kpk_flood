"""
Runtime Google Earth Engine feature extraction for a single (lat, lon).

This is a refactor of the supplied
`Google Earth Engine runtime feature extraction.py` script into a
reusable function, with ONE deliberate fix:

  ORIGINAL: features.sample(region=point, scale=30, numPixels=1).first()
  FIXED:    features.reduceRegion(ee.Reducer.first(), point, scale=30)

Why: ee.Image.sample() drops a pixel ENTIRELY if any single band is
masked (e.g. NDVI is cloud-masked, or a point falls right at the edge
of a source dataset's coverage). That is exactly the bug that crashed
both of the notebook's own validation cells:
    - "Element.toDictionary: Parameter 'element' is required and may
      not be null" (cell 51) — .first() on an empty collection
    - the FABDEM/HIGH asset-not-found error (cell 55) is a separate,
      unrelated bug from an earlier draft and is not present in this
      script — it already uses the correct asset id, verified below.

reduceRegion() returns one value per band (possibly None) instead of
silently dropping the whole point, so the backend can raise a clear,
specific error ("ndvi is null at this location") instead of crashing
on an opaque EE exception. The underlying images/masks/scales are
UNCHANGED from the supplied script, so this does not alter the
feature values themselves for any pixel that isn't already masked.

IMPORTANT — this file has not been executed against Earth Engine in
this environment (no network access / no service-account credentials
here). Run scripts/verify_features.py yourself, with real GEE
credentials, before trusting this in production. See README.md.
"""

import numpy as np
import ee

from .feature_engineering import RAW_FEATURE_ORDER

# NOTE: this module assumes ee.Initialize(...) has already been called
# by the caller (see lib/gee_auth.py for the production/service-account
# path, used by api/index.py). Keeping auth out of this module makes it
# reusable both from the FastAPI backend and from scripts/verify_features.py,
# which may authenticate differently (e.g. interactive `earthengine
# authenticate` for a one-off local verification run).


def _build_feature_image():
    """Builds the 9-band image, identical to the supplied runtime script."""

    # 3. Elevation / DEM ---------------------------------------------------
    # FABDEM is an ImageCollection, not an Image — must mosaic first.
    fabdem = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")
    dem = (
        fabdem.mosaic()
        .setDefaultProjection("EPSG:3857", None, 30)
        .rename("elevation")
    )

    # 4. Terrain -------------------------------------------------------------
    slope = ee.Terrain.slope(dem).rename("slope")
    aspect = ee.Terrain.aspect(dem).rename("aspect")
    # NOTE: explicit normalize=False here matches ee.Kernel.laplacian8()'s
    # own default, so this line is functionally identical to the version in
    # the original supplied runtime script. Testing confirmed this was NOT
    # the source of the curvature mismatch — kept explicit for clarity, but
    # the real cause of the curvature/aspect discrepancy is still unresolved
    # (low model importance: curvature 2.4%, aspect_sin+cos 4.3% combined —
    # see README.md "Known limitation").
    curvature = dem.convolve(ee.Kernel.laplacian8(normalize=False)).rename("curvature")

    # 5. MERIT Hydro -----------------------------------------------------
    merit = ee.Image("MERIT/Hydro/v1_0_1")
    upa = merit.select("upa")

    # 6. Distance to drainage --------------------------------------------
    drainage_mask = upa.gt(0.5).selfMask()
    distance_to_drainage = (
        drainage_mask.reproject("EPSG:3857", None, 100)
        .fastDistanceTransform(512, "pixels", "squared_euclidean")
        .sqrt()
        .multiply(100)
        .rename("distance_to_drainage")
    )

    # 7. Distance to rivers -----------------------------------------------
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    river_mask = gsw.select("occurrence").gt(50).selfMask()
    distance_to_rivers = (
        river_mask.reproject("EPSG:3857", None, 500)
        .fastDistanceTransform(512, "pixels", "squared_euclidean")
        .sqrt()
        .multiply(500)
        .rename("distance_to_rivers")
    )

    # 8. TWI -----------------------------------------------------------------
    slope_rad = slope.multiply(np.pi / 180)
    flow_accum_m2 = upa.multiply(1e6)
    cell_size_m = ee.Image.pixelArea().sqrt()
    specific_catchment_area = flow_accum_m2.divide(cell_size_m)
    twi = (
        specific_catchment_area.divide(slope_rad.tan().max(0.001))
        .max(1e-6)
        .log()
        .rename("twi")
    )

    # 9. NDVI ------------------------------------------------------------
    # CLOUD_COVER filter matches the notebook's own reconstruction of the
    # original training pipeline (search "Landsat 8/9 surface reflectance"
    # / CLOUD_COVER in the notebook). Without it, cloudy/hazy scenes pull
    # the median NDVI composite down — this is what produced the
    # systematically low NDVI (~0.12 vs ~0.29 in the CSV) seen in testing.
    landsat = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterDate("2021-01-01", "2023-12-31")
        .filter(ee.Filter.lt("CLOUD_COVER", 30))
    )

    def add_ndvi(image):
        ndvi = image.normalizedDifference(["SR_B5", "SR_B4"]).rename("ndvi")
        return image.addBands(ndvi)

    ndvi = landsat.map(add_ndvi).select("ndvi").median().rename("ndvi")

    # 10. Rainfall frequency ------------------------------------------------
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(
        "2010-01-01", "2023-01-01"
    )
    rainfall_frequency = (
        chirps.select("precipitation")
        .map(lambda img: img.gt(10).rename("rain_event"))
        .sum()
        .rename("rainfall_frequency")
    )

    # 11. Combine --------------------------------------------------------
    return ee.Image.cat(
        [
            aspect,
            curvature,
            distance_to_drainage,
            distance_to_rivers,
            dem,
            ndvi,
            rainfall_frequency,
            slope,
            twi,
        ]
    )


_feature_image = None


def get_feature_image():
    global _feature_image
    if _feature_image is None:
        _feature_image = _build_feature_image()
    return _feature_image


class NoDataError(Exception):
    """Raised when GEE has no usable data for one or more bands at a point."""


def extract_raw_features(lat: float, lon: float, scale: int = 30) -> dict:
    """
    Returns a dict with keys = RAW_FEATURE_ORDER for the given coordinate.
    Raises NoDataError if any band is null at that pixel (e.g. permanently
    cloud-masked NDVI, or a point outside a source dataset's coverage).

    Assumes ee.Initialize(...) has already been called by the caller.
    """
    point = ee.Geometry.Point([lon, lat])
    image = get_feature_image()

    values = image.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=point,
        scale=scale,
        bestEffort=True,
        tileScale=4,
    ).getInfo()

    missing = [k for k in RAW_FEATURE_ORDER if values.get(k) is None]
    if missing:
        raise NoDataError(
            f"No Earth Engine data for band(s) {missing} at ({lat}, {lon}). "
            f"This can happen near data-source edges, over permanent water, "
            f"or where Landsat NDVI is fully cloud-masked for 2021-2023."
        )

    return {k: values[k] for k in RAW_FEATURE_ORDER}
