# Atmospheric Rivers and Avalanches

## Background
This codebase will download [ERA5 6 hourly pressure level data](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview) from 1993-2023 in the vicinity of SE Alaska and apply an atmospheric river (AR) detection algorithm. Outputs will include a datacube of AR objects, a table of descriptive statistics about those AR objects, and a geodataframe for future work correlating avalanche events to ARs. 

The AR detection algorithm used here is adapted from [Guan & Waliser (2015)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2015JD024257) and uses a combination of vertically integrated water vapor transport (IVT), geometric shape, and directional criteria to define ARs. See the annotated bibilography document for more detail and other references. Users of this codebase should know the following information about atmospheric river criteria:

All of the AR criteria proposed by Guan and Waliser (2015) either judge the geometry of the candidate AR object and/or measure some parameter of integrated water vapor transport (magnitude, direction, and poleward component). Each candidate AR object represents a specific time (six hour resolution) slice of the IVT data where contiguous groups of grid cells in which the magntiude the target quantile are labeled with unqiue integers >=1. The 0 value is reserved to mask the region that did not exceed the IVT quantile threshold.

The five critera are:

### Length
Objects longer than 2000 km are retained as AR candidates.
### Length/Width Ratio
The ratio of the major to minor axes of the ellipse fit to the AR object shape. Objects with length/width ratio greater than 2 are retained as AR candidates.
### Coherence in IVT Direction
If more than half of the grid cells have IVT deviating more than 45° from the object's mean IVT, the object is discarded. This is aimed to filter out objects which do not exhibit a coherent IVT direction and to make a calculation of mean/characteristic IVT direction physically meaningful.
### Object Mean Meridional IVT
Considering the notion that ARs transport moisture from low to high latitudes, an object is discarded if the mean IVT does not have an appreciable poleward component (>50 kg m−1 s−1).
### Consistency Between Object Mean IVT Direction and Overall Orientation
The overall orientation of the object (i.e., the direction of the shape elongation) is the mean azimuth of the arc connecting the two boundary grid cells with the maximum great circle distance. An object is discarded if the direction of mean IVT deviates from the overall orientation by more than 45°. This is aimed to filter objects where the IVT does not transport in the direction of object elongation.

## Structure
 - All download specifications and model parameters are defined in `config.py`.
 - The `download.py` script will download the necessary ERA5 input data.
 - The `compute_ivt.py` script will transform the downloaded ERA5 input data into a datacube with the additional variables of IVT magnitude, direction, and an IVT quantile value.
 - The `ar_detection.py` module contains a collection of AR detection functions that will filter AR candidates based on the criteria and create a shapefile output of objects classified as ARs. These functions may be orchestrated from a notebook (see `AR_detection.ipynb`) or ran as a script. 

## Usage
1. Register for a Climate Data Store (CDS) account and install the CDS API client according to the instructions [here].(https://cds.climate.copernicus.eu/api-how-to). Be sure to accept the user agreement. 
2. Create a new conda environment using the `environment.yml` file in this codebase.
3. Set a local environment variable defining the data directory. If the directory does not yet exist, `config.py` will create it, e.g. `export AR_DATA_DIR=path/to/store/ar/data`
5. Review parameters in `config.py` and adjust if desired. Note that there is a download request limit of 120,000 items, so adjusting the timestep or date range may overload the request and break the script.
6. Execute `download.py`
7. Execute `compute_ivt.py`
8. Execute `detect_ars.py` or use the `AR_detection.ipynb` to orchestrate the detection.