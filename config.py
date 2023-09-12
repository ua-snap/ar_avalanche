"""Atmospheric Rivers and Avalanches Configuration

This config file is imported by other scripts and notebooks in this codebase. Settings include data input/output locations based on user env variables,
ERA5 download parameters (variables, time interval, bounding box), and parameters used in vapor transport quantile computations and atmospheric river detection algorithms."""

import os
from pathlib import Path

# get user env variable
DATA_DIR = Path(os.getenv("AR_DATA_DIR"))
DATA_DIR.mkdir(exist_ok=True, parents=True)

# directory for ERA5 downloads
DOWNLOAD_DIR = DATA_DIR.joinpath("inputs")
DOWNLOAD_DIR.mkdir(exist_ok=True)

# directory for outputs
OUTPUT_DIR = DATA_DIR.joinpath("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# path for IVT computation and AR detection input file
ard_fp = DOWNLOAD_DIR.joinpath("ar_detection_inputs.nc")

# path for AR detection results shapefile
shp_fp = OUTPUT_DIR.joinpath("detected_ars.shp")

# path for shapefile column name crosswalk csv
csv_fp = OUTPUT_DIR.joinpath("detected_ars.csv")

# path for landfall ARs results shapefile
landfall_shp = OUTPUT_DIR.joinpath("landfall_ars.shp")

# path for landfall ARs results column name crosswalk csv
landfall_csv = OUTPUT_DIR.joinpath("landfall_ars.csv")

# path for landfall AR events results shapefile
landfall_events_shp = OUTPUT_DIR.joinpath("landfall_ar_events.shp")

# path for landfall AR events results column name crosswalk csv
landfall_events_csv = OUTPUT_DIR.joinpath("landfall_ar_events.csv")

# path for AR event coastal impact results shapefile
coastal_impact_shp = OUTPUT_DIR.joinpath("landfall_ar_events_coastal_impact.shp")

# path for AR event coastal impact results column name crosswalk csv
coastal_impact_csv = OUTPUT_DIR.joinpath("landfall_ar_events_coastal_impact.csv")

# path for log CSV
log_fp = OUTPUT_DIR.joinpath("log.csv")

# path to Alaska coastline shapefile (included in Github repo tree)
ak_shp = Path("./shp/Alaska_Coast_Simplified_Polygon.shp")

# path to Canada coastline shapefile (included in Github repo tree)
ca_shp = Path("./shp/gpr_000b11a_e.shp")

# path to Alaska point location table (included in Github repo tree)
ak_pts = Path("./tbl/alaska_point_locations.csv")

# path to revised avalanche database (included in Github repo tree)
avy_db = Path("./tbl/avy_db_rev.csv")

# path to ENSO table (included in Github repo tree)
enso_csv = Path("./tbl/enso_1980-2021.csv")

# path to ENSO table (included in Github repo tree)
pdo_csv = Path("./tbl/pdo_1980-2021.csv")

# ERA5 download parameters
dataset = "reanalysis-era5-single-levels"
varnames = [
    "vertical_integral_of_eastward_water_vapour_flux",
    "vertical_integral_of_northward_water_vapour_flux",
]

# list of paths for ERA5 .nc downloads
era5_fps = [DOWNLOAD_DIR.joinpath(str(v + ".nc")) for v in varnames]

# path for concatenated ERA5 .nc
era5_merged = DOWNLOAD_DIR.joinpath("era5_merged.nc")

# use reduced time spans, smaller bboxes for testing
start_year = 1980
end_year = 2022
bbox = [10, -179, 66, -120]

# download options
era5_kwargs = {
    "product_type": "reanalysis",
    "format": "netcdf",
    "year": [str(year) for year in range(start_year, end_year)],
    "month": [str(month).zfill(2) for month in range(1, 13)],
    "day": [str(day).zfill(2) for day in range(1, 32)],
    "time": [
        "0" + str(x) + ":00" if x <= 9 else str(x) + ":00" for x in range(0, 24, 6)
    ],
    "area": bbox,
}


# confirm CDS API credentials are in place prior to ERA5 download
def api_credentials_check():
    cds_api_prompt = "Climate Data Store API credentials were not found in your $HOME directory. Please verify and store a valid API key in a .cdsapirc file and visit https://cds.climate.copernicus.eu/api-how-to#install-the-cds-api-key for instructions."
    assert ".cdsapirc" in os.listdir(os.environ["HOME"]), cds_api_prompt


# AR detection parameters, defaults identical to Guan & Waliser 2015
ar_params = {
    # days before/after to compute IVT percentile (e.g., 75 before and 75 after for a ~5-month window
    "window": 75,
    # threshold percentile to identify high IVT instances
    "ivt_percentile": 90,
    # secondary IVT criteria for identifying ARs in polar low moisture areas; units are kg m**−1 s**−1 (same as IVT)
    "ivt_floor": 100,
    # criteria for directional coherence; compare this value against mean IVT direction; units are degrees
    "direction_deviation_threshold": 45,
    # criteria for determining poleward component of moisture; units are kg m**−1 s**−1 (same as IVT)
    "mean_meridional": 50,
    # criteria for direction/orientation coherence; compare this value against mean IVT direction; units are degrees
    "orientation_deviation_threshold": 45,
    # geometric criteria for AR object axis; used to calculate width and length-to-width ratio; units are km
    "min_axis_length": 2000,
}
