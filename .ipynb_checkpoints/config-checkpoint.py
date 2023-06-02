# this config file is designed to be called by other scripts in this codebase. This file will:

# define data input/output location(s) based on user env variables
# define parameters used in the era5 download processing (eg, variables, pressure levels, time interval, bounding box, etc)
# define parameters used in the calculation of IVT (eg intensity percentile, percentile window)
# define a function to check for CDS API credentials

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

# filepath for era5 .nc download
era5_fp = DOWNLOAD_DIR.joinpath("era5_ivt_params.nc")

# filepath for IVT computation / AR detection input .nc
ard_fp = DOWNLOAD_DIR.joinpath("ar_detection_inputs.nc")

# era5 download parameters
dataset = 'reanalysis-era5-single-levels'
varnames = ['vertical_integral_of_eastward_water_vapour_flux',
           'vertical_integral_of_northward_water_vapour_flux'
           ]
start_year = 1993
end_year = 2023
bbox = [60, -150, 50, -130]

era5_kwargs = {
    "variable": varnames,
    "product_type": "reanalysis",
    "format": "netcdf",
    "year": [str(year) for year in range(start_year, end_year)],
    "month": [str(month).zfill(2) for month in range(1, 13)],
    "day": [str(day).zfill(2) for day in range(1, 32)],
    "time": ["0" + str(x) + ":00" if x <= 9 else str(x) + ":00" for x in range(0, 24, 6)],
    "area": bbox
}



# AR detection parameters
# these are identical to Guan & Waliser 2015 (see README.md)
    # window = days before / after to compute IVT percentile (eg, 60 = 60d before and 60d after for an approximate 4-month window
    # ivt_percentile = threshold percentile to identify high IVT instances
    # ivt_floor = secondary IVT criteria for identifying ARs in polar low moisture areas; units are kg m**−1 s**−1 (same as IVT)
    # direction_deviation_threshold = criteria for directional coherence; compare this value against mean IVT direction; units are degrees
    # mean_meridional = criteria for determining poleward component of moisture; units are kg m**−1 s**−1 (same as IVT)
    # orientation_deviation_threshold = criteria for direction/orientation coherence; compare this value against mean IVT direction; units are degrees
    # min_axis_length = geometric criteria for AR object axis; used to calculate width and length-to-width ratio; units are km

ar_params = {
    "window": 75,
    "ivt_percentile": 85,
    "ivt_floor": 100,
    "direction_deviation_threshold": 45,
    "mean_meridional": 50,
    "orientation_deviation_threshold": 45,
    "min_axis_length": 1000
}


# CDS API credentials check function
# delivers a message before a CDS API error prints
def api_credentials_check():
    cds_api_prompt = "Climate Data Store API credentials were not found in your $HOME directory. Please verify and store a valid API key in a .cdsapirc file and visit https://cds.climate.copernicus.eu/api-how-to#install-the-cds-api-key for instructions."
    assert ".cdsapirc" in os.listdir(os.environ["HOME"]), cds_api_prompt