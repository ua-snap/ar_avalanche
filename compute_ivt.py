"""Integrated Vapor Transport (IVT)

This module computes integrated vapor transport (IVT) magnitude and direction from two ERA5 single-level variables of "vertical integral of eastward water vapour flux" and "vertical integral of northward water vapour flux". A target IVT magnitude percentile (e.g., 85th) value is also computed on a day-of-year (DOY) basis. For each DOY, the target percentile is computed for all grid cells using the IVT magnitude values within some time window (e.g., 5 months) centered on that DOY using data from all years in the period of record to compute the percentiles. The variables of IVT magnitude, IVT direction, and IVT magntiude percentile are composed as an xarray DataSet and written to disk as a netCDF file. The size of the time window and the target percentile are set by config.py.

Example Usage:
    You can use the module from the command line with no arguements, like so:
        $ python compute_ivt.py
    Or import these functions in a Jupyter Notebook to experiment with other paramter values and/or orchestrate with a Dask client.

"""
import xarray as xr
import numpy as np
from tqdm import tqdm

from config import era5_fp, ar_params, ard_fp


def compute_magnitude(u, v):
    """Compute vapor transport magnitude from eastward (u) and northward (v) components.These may be considered two legs of a right triangle. This magnitude computation is similar to how wind speed is computed from u and v components.

    Parameters
    u : xarray.DataArray
        ERA5 eastward vapor flux
    v : xarray.DataArray
        ERA5 northward vapor flux

    Returns
    -------
    xarray.DataArray
        Total magnitude of vapor transport
    """
    return xr.apply_ufunc(np.hypot, u, v, dask='parallelized')


def compute_direction(u, v):
    """Compute vapor transport direction from eastward (u) and northward (v) components. reference: https://www.eol.ucar.edu/content/wind-direction-quick-reference

    Parameters
    u : xarray.DataArray
        ERA5 eastward vapor flux
    v : xarray.DataArray
        ERA5 northward vapor flux

    Returns
    -------
    xarray.DataArray
        direction that the the vapor transport is coming from in degrees with respect to true north (0=north, 90=east, 180=south, 270=west)
    """
    func = lambda x, y: 270 - ((180 / np.pi) * np.arctan2(x, y))
    return xr.apply_ufunc(func, u, v, dask='parallelized')


def generate_start_end_doys(day_of_year, window):
    """Generate start and end DOYs for a time window centered on a particular DOY.

    Parameters
    ----------
    day_of_year : (int)
        DOY for which to center the time window
    window : (int)
        time window length in days

    Returns
    -------
    tuple
        two-tuple of integer DOYs representing the start and end of the time window 
    """
    half_window = window // 2
    # check leap year
    is_leap_year = (day_of_year == 366)
    total_days = 366 if is_leap_year else 365
    
    start_day_of_year = (day_of_year - half_window) % total_days
    end_day_of_year = (day_of_year + half_window) % total_days
    
    return start_day_of_year, end_day_of_year


def compute_period_of_record_quantile(da, day_of_year, window, target_quantile):
    """Compute a single quantile value for IVT magnitude for all grid cells for specific DOY periods.

    Parameters
    ----------
    da : xarray.DataArray
        IVT mangtiude
    day_of_year : int
        DOY to compute target percentile for
    window : int
        Time window length in days
    target_quantile : float
        Quantile (between 0 and 1) to compute. For some vector V, the q-th quantile of V is the value q of the way from the minimum to the maximum in a sorted copy of V. Note that in the NumPy implementation, the `quantile` and `percentile` functions are equivalent: a quantile value of 0.85 is equivalent to the 85th percentile.

    Returns
    -------
    result
        xarray.DataArray containing the q-th quantile of IVT magnitude for the given DOY.
    """
    leap_year = (day_of_year == 366)
    
    start_day_of_year, end_day_of_year = generate_start_end_doys(day_of_year, window)
    
    if day_of_year >= window // 2 and 365 - day_of_year >= window // 2:
        # subset without wrapping the year boundary (e.g., DOY is 180)
        subset = da.sel(time=((da.doy >= start_day_of_year) & (da.doy <= end_day_of_year)))
    else:
        # subset with wrapping the year boundary (e.g., DOY is 360, window is 30 days)
        subset = da.sel(time=((da.doy >= start_day_of_year) | (da.doy <= end_day_of_year)))

    # for DOYs other than 366, omit values from DOY 366 from the results
    if not leap_year:
        mask = subset['doy'] != 366
        subset = subset.where(mask, drop=True)

    result = subset.reduce(np.nanquantile, q=target_quantile, dim="time")
    return result


def compute_quantiles_for_doy_range(da, doy_start, doy_stop, window, target_quantile):
    """Compute quantiles for a range of target DOYs and return them in a DataArray where each slice in time contains an array of target quantile IVT magnitude values for all grid cells.

    Parameters
    ----------
     da : xarray.DataArray
        IVT mangtiude
    doy_start : int
        Start of DOY range (min value is 1)
    doy_stop : int
        End of DOY range (max value is 367)
    window : int
        Time window length in days used to compute quantile values for each individual DOY
    target_quantile : float
        Quantile (between 0 and 1) to compute.

    Returns
    -------
    combined_da
        xarray.DataArray containing the q-th quantiles of IVT magnitude for all DOYs within the given DOY range.
    """
    quantile_results = []
    for day_of_year in tqdm(range(doy_start, doy_stop)):
        doy_quantile = compute_period_of_record_quantile(da, day_of_year, window, target_quantile)
        quantile_results.append(doy_quantile)

    for i, quantile_result in enumerate(quantile_results):
        doy_result = quantile_result.assign_coords(doy=i + 1)
        quantile_results[i] = doy_result

    # concatenate list of DataArrays along the 'doy' dimension
    combined_da = xr.concat(quantile_results, dim='doy')
    return combined_da


def compute_full_ivt_datacube(era5_fp, ar_params, ard_fp):
    """Open the downloaded ERA5 dataset, compute IVT magnitude, direction, quantile and write results to new .nc file to use as input for AR detection.

    Parameters
    ----------
    era5_fp : pathlib.Path
        Path to downloaded input ERA5 data
    ar_params : dict
        Guan & Waliser (2015) inspired parameters
    ard_fp : pathlib.Path
        Path to write results to disk

    Returns
    -------
    None
    """
    with xr.open_dataset(era5_fp) as ds:
        # add DOY coords to input dataset
        dsc = ds.assign_coords(doy=ds.time.dt.dayofyear)
        # p71.162 / p72.162 codes for eastward "u" / northward "v" components
        dsc["ivt_mag"] = compute_magnitude(dsc["p71.162"], dsc["p72.162"]).astype(int)
        dsc["ivt_dir"] = compute_direction(dsc["p71.162"], dsc["p72.162"]).astype(int) 

        # CP note: leaving this for potential future optimization
        # chunk to avoid memory error
        # ds = ds.chunk({"latitude": 1, "longitude": 1})

        time_window_days = ar_params["window"] * 2
        qth_target_quantile = ar_params["ivt_percentile"] / 100
        
        dsc["ivt_quantile"] = compute_quantiles_for_doy_range(dsc["ivt_mag"], 1, 367, time_window_days, qth_target_quantile)
        # the raw east / north components are not needed in the output
        dsc = dsc.drop_vars(["p71.162", "p72.162"])
        # write to disk
        dsc.to_netcdf(ard_fp)


if __name__ == "__main__":
    
    print(f"Using {era5_fp} to compute IVT magntiude, direction, and the {ar_params['ivt_percentile'] / 100} IVT quantile for a time window of {ar_params['window'] * 2} days centered on each day-of-year.")
    
    compute_full_ivt_datacube(era5_fp, ar_params, ard_fp)
    
    print(f"Computation complete! AR detection parameters saved to {ard_fp}")
