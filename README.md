# Atmospheric Rivers and Avalanches

## Background
This codebase will download [ERA5 6 hourly pressure level data](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview) from 1993-2023 in the vicinity of SE Alaska and apply an atmospheric river (AR) detection algorithm. Outputs will include a datacube of AR objects, a table of descriptive statistics about those AR objects, and a geodataframe for future work correlating avalanche events to ARs. 

The AR detection algorithm used here is adapted from [Guan & Waliser (2015)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2015JD024257) and uses a combination of vertically integrated water vapor transport (IVT), geometric shape, and directional criteria to define ARs. See the annotated bibilography document for more detail and other references.


## Structure
 - All download specifications and model parameters are defined in `config.py`.
 - The `download.py` script will download the necessary ERA5 input data.
 - The `compute_ivt.py` script will transform the downloaded ERA5 input data into a datacube with the additional variables of IVT magnitude and direction.
 - The `ar_detection.py` module contains a collection of AR detection functions designed to be run interactively from a Jupyter notebook.

## Usage

1. Register for a Climate Data Store (CDS) account and install the CDS API client according to the instructions [here].(https://cds.climate.copernicus.eu/api-how-to). Be sure to accept the user agreement. 
2. Create a new conda environment using the `environment.yml` file in this codebase.
3. Set a local environment variable defining the data directory. If the directory does not yet exist, `config.py` will create it, e.g. `export AR_DATA_DIR=path/to/store/ar/data`
5. Review parameters in `config.py` and adjust if desired. Note that there is a download request limit of 120,000 items, so adjusting the timestep or date range may overload the request and break the script.
6. Execute `download.py`
7. Execute `compute_ivt.py`
8. Use the `AR_detection_testing.ipynb` notebook to run the detection functions and check results against known AR events.