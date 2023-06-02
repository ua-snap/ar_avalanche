## Atmospheric Rivers and Avalanches

**Note: This project is currently unfinished as of 6/2/2023. The ```download.py``` and ```compute_ivt.py``` scripts are in working condition, but additional functions need to be developed in ```ar_detection.py``` to apply all AR detection criteria and output final AR statistics.**

This codebase will download [ERA5 6 hourly pressure level data](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview) from 1993-2023 in the vicinity of SE Alaska and apply an atmospheric river (AR) detection algorithm. Outputs will include a datacube of AR objects, a table of descriptive statistics about those AR objects, and a geodataframe for future work correlating avalanche events to ARs. 


The AR detection algorithm used here is adapted from [Guan & Waliser (2015)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2015JD024257) and uses a combination of vertically integrated water vapor transport (IVT), geometric shape, and directional criteria to define ARs.


All download specifications and model parameters are defined in ```config.py```. The scripts ```download.py``` and ```compute_ivt.py``` are designed to be run from the command line with no arguments. The ```ar_detection.py``` script contains a collection of AR detection functions designed to be run interactively from a Jupyter notebook. Time slices can be selected within this notebook to focus on known AR events and test the model.


#### To run this code:

1. Register for a Climate Data Store (CDS) account and install the CDS API client according to the instructions [here].(https://cds.climate.copernicus.eu/api-how-to). Be sure to accept the user agreement. 

2. Create a new conda environment using the ```environment.yml``` file in this codebase.

3. Set a local environment variable defining the data directory. If the directory does not yet exist, the ```config.py``` script will create it. For example:
    - export AR_DATA_DIR=path/to/store/ar/data  
\
4. Adjust parameters in the ```config.py``` if necessary. Note that there is a download request limit of 120,000 items, so adjusting the timestep or date range may overload the request and break the script.
5. Execute ```download.py``` and ```compute_ivt.py``` from the command line with no arguments.
6. Use the ```AR_detection_testing.ipynb``` notebook to run the detection functions and check results against known AR events.