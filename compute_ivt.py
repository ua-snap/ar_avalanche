# this computation script is designed to be run from the command line with no arguments. 

# it will read parameters from the config.py file (eg input/outputs, ar detection parameters),
# compute ar detection data cube,
# save ar detection data cube to output directory as .nc file

import xarray as xr
import numpy as np
from config import era5_fp, ar_params, ard_fp


# function to compute IVT magnitude from eastward and northward components
# output is kg m**2/s**2
# see here for math reference: https://www.eol.ucar.edu/content/wind-direction-quick-reference
def magnitude(a, b):
    func = lambda x, y: np.sqrt(x**2 + y**2)
    return xr.apply_ufunc(func, a, b, dask='parallelized')

# function to compute geographic IVT direction from eastward and northward components
# this is the degrees with respect to true north (0=north,90=east,180=south,270=west) that the wind is coming FROM
# output is in degrees
# see here for math reference: https://www.eol.ucar.edu/content/wind-direction-quick-reference
def direction(a, b):
    func = lambda x, y: 270 - ((180/np.pi) * np.arctan2(x, y))
    return xr.apply_ufunc(func, a, b, dask='parallelized')

# function to compute IVT magnitude percentile on a rolling time window
# this function takes a long time to run and require lots of memory!
# takes hours even with ivt_mag as integer and when run chunk by chunk with map_blocks()
# note that without using map_blocks(), there will definitely be a memory error
def rolling_pctile(da, window, percentile):
    r = da.rolling(time=window, center=True)
    pctile = r.reduce(np.quantile, q=percentile)
    return pctile


# function to compute IVT magnitude percentile over DOY window subset of the entire time domain 
# (ie, percentile of window over 'normal' period)
# this function takes a long time to run and require lots of memory!
# takes hours even with ivt_mag as integer and when run chunk by chunk with map_blocks()
# note that without using map_blocks(), there will definitely be a memory error
def normal_pctile(da, window, percentile):
    
    # create dictionary with all DOYs (1-366, to include leap years) as key and list of DOYs in window as values
    # there will be one fewer day in the window on a leap year, should not really matter much
    doy_dict = {}
    for i in range(1, 367):
        window_doys = [*range((i - window),(i + window))]
        doys_to_drop = []
        for w in window_doys:
            if w<1:
                window_doys.append(365+w)
                doys_to_drop.append(w)
            elif w>365:
                window_doys.append(w-365)
                doys_to_drop.append(w)
            else:
                pass   
        doy_dict[i] = [k for k in window_doys if k not in doys_to_drop]
    
    # add day of year (doy) as coordinate label for time dimension in the input ivt_mag data array
    # !! moved to main compute_ivt() function !!
    # da = da.assign_coords(doy=da.time.dt.dayofyear)
    
    # copy ivt_mag data array to new da_ variable, and also:
    # rename the variable
    # assign all values as NA
    # this is now a template data array with same size/dimensions as the input to store percentile results
    
    da_ = da.copy(deep=False).rename('ivt_pctile_n').where(da.values == -9999)
    
    # loop thru input ivt_mag data array over time dimension, subsetting by doy window list
    # compute the ivt_mag percentile over the entire time dimension of the subset
    # assign the new ivt mag percentile array values to the proper timestep in the empty template array

    for a in range(len(da.time)):
        sub_da = da.where(da.doy.isin(doy_dict[int(da[a].doy)])) 
        pa = sub_da.quantile(percentile, dim='time', skipna=True)
        da_[a] = pa
 
    return da_


# function to open era5 dataset and compute IVT magnitude, IVT direction, and rolling window IVT magnitude percentile
# saves to new .nc file to use as input for AR detection notebook
def compute_ivt(era5_fp, ar_params, ard_fp):
    
    with xr.open_dataset(era5_fp) as ds:
        
        #add DOY coords to input dataset
        ds = ds.assign_coords(doy=ds.time.dt.dayofyear)

        # chunk to avoid memory error
        ds = ds.chunk({"latitude": 1, "longitude": 1})

        # p71.162 = code eastward "u" component
        # p72.162 = code for northward "v" component
        ds['ivt_mag'] = magnitude(ds['p71.162'], ds['p72.162']).astype(int)
        ds['ivt_dir'] = direction(ds['p71.162'], ds['p72.162']).astype(int)

        # for seasonal percentile, recall this is 6hr data, so there are 4 timesteps for each day
        # so the rolling window = number of days * 4 * 2 (forward and backwards)
        # for normal percentile, use number of days as window... the DOY dictionary function will apply forward and backward
        w = ar_params['window']*8
        q = ar_params['ivt_percentile']/100
        ds['ivt_pctile_s'] = ds['ivt_mag'].map_blocks(rolling_pctile, kwargs={"window": w, "percentile" : q}, template=ds['ivt_mag']).compute()
        ds['ivt_pctile_n'] = ds['ivt_mag'].map_blocks(normal_pctile, kwargs={"window": ar_params['window'], "percentile" : q}, template=ds['ivt_mag']).compute()

        ds.to_netcdf(ard_fp)
    
    return ard_fp




if __name__ == "__main__":
    
    print("Computing IVT and AR detection inputs from " + str(era5_fp) + " using these parameters: ")
    print(ar_params)
    
    fp = compute_ivt(era5_fp, ar_params, ard_fp)
    
    if Path(fp).exists:
        print("Computation complete! AR detection parameters saved to: " + str(fp))
    else:
        print("Expected output not found, computation may have failed.")