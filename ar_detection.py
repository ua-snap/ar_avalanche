# this AR detection script will store a set of functions designed to be run from a notebook.
import xarray as xr
import numpy as np
import rasterio
from scipy.ndimage import label, generate_binary_structure, labeled_comprehension
from skimage.measure import regionprops
from scipy.stats import circmean

from config import ar_params, spatial_resolution_reprojected


def compute_intensity_mask(ivt_mag, ivt_quantile, ivt_floor):
    """Compute IVT mask where IVT magnitude exceeds quantile and floor values.

    Parameters
    ----------
    ivt_mag : xarray.DataArray
        IVT magntiude
    ivt_quantile : xarray.DataArray
        IVT quantile values used for thresholding
    ivt_floor : int
        Minimum IVT value for threshold consideration

    Returns
    -------
    xarray.DataArray
        IVT magnitude values where the magnitude exceeds the quantile and the IVT floor. Else, zero.
    """
    func = lambda x, y, z: xr.where((x > y) & (x > z), x, 0)
    return xr.apply_ufunc(func, ivt_mag, ivt_quantile, ivt_floor, dask='parallelized')


def label_contiguous_mask_regions(ivt_mask):
    """Label contiguous geometric regions of IVT values that exceed the targent quantile and floor for each time step. Each region is labeled with a unique integer identifier. Output will have the same dimensions as the IVT intensity mask     input and be reprojected to 3338 with a prescribed grid size in meters.

    Parameters
    ----------
    ivt_mask : xarray.DataArray
        IVT magnitude where the valuyes exceed the quantile and IVT floor.

    Returns
    -------
    xarray.DataArray
        Labeled (with unique integers) discrete and contiguous regions of IVT threshold exceedance.
    """
    # set search structure to include diagonal neighbors
    s = generate_binary_structure(2,2)
    
    # intialize the output by copying ivt_mask, renaming it, and setting all values to a known no data value
    # this constructs a template DataArray with the same size/dimensions as the input to store region labeling results
    da = ivt_mask.copy(deep=False).rename('regions').where(ivt_mask.values == -9999)
    
    # CP note: looping through the time dimension is likely a chokepoint
    # this is a good candidate for parallelization
    for a in range(len(ivt_mask.time)):
        # label contiguous regions of each timestep
        labeled_array, num_features = label(ivt_mask[a].values, structure=s)
        # map labeled regions to the timestep in the template array
        da[a] = labeled_array

    # prescribe CRS
    da.rio.write_crs("epsg:4326", inplace=True)
    # reproject to 3338 with prescribed grid cell size, set NoData values to 0
    da_3338 = da.rio.reproject("epsg:3338", resolution=spatial_resolution_reprojected,
                               nodata=0, resampling=rasterio.enums.Resampling.nearest)
    # return a categorical raster where all non-zero values are region labels
    return da_3338


def filter_regions_by_geometry(regions, min_axis_length):
    """Modify the labeled regions DataArray in place by removing regions not meeting AR shape criteria.
    Function requires the entire spatial domain for shape measurement, so dask chunking along lat/lon dimensions should be avoided.
    Regions not meeting shape criteria will be added to a "drop" dictionary using the timestep as a key.

    Parameters
    ----------
    regions : xarray.DataArray
        labeled regions of contiguous IVT quantile and floor exceedance with time, lat, and lon coordinates
    min_axis_length : int
        units in km

    Returns
    -------
    drop_dict
        dict of regions failing to meet AR shape criteria where keys are timesteps
        and values are the labeled regions of the time slice that fail to meet the criteria.
        the output dictionary is just a reference for confirming that the non-AR labels are dropped.
    """

    spatial_resolution_reprojected_km = spatial_resolution_reprojected / 1000
    min_axis_length_pixels = min_axis_length / spatial_resolution_reprojected_km

    drop_dict = {}
    
    for labeled_time_slice in regions:
        
        props = regionprops(labeled_time_slice.astype(int).values)

        drop_list = []
        for p in props:
            # check axis length criteria
            if p.major_axis_length < min_axis_length_pixels:
                drop_list.append(p.label)
            # check length to width ratio 2:1 or greater criteria  
            elif (p.major_axis_length / p.minor_axis_length) < 2:
                drop_list.append(p.label)
        
        if len(drop_list) > 0:
            drop_dict[labeled_time_slice.time.values] = drop_list
    
    # use the drop dictionary to do another loop thru the original dataset and reassign dropped labels to 0
    for d in drop_dict:
        regions.loc[dict(time=d)] = xr.where(regions.sel(time=d).isin(drop_dict[d]), 0, regions.sel(time=d))
         
    return drop_dict


def is_directionally_coherent(ivt_direction_values,
                              range_degrees=ar_params["direction_deviation_threshold"]):
    """
    Determine IVT directional coherence. If more than half of the grid cells have IVT directions deviating
    greater than the threshold from the object's mean IVT, then then the object is not directionally coherent.

    Parameters:
        ivt_direction_values (np.ndarray): The IVT direction values to check.
        range_deg (int): The range in degrees.

    Returns:
        int: 0 or 1 expression of IVT directional coherence
    """
    mean_reference_deg = circmean(ivt_direction_values, high=360, low=0)
    
    degrees_deviation = abs((ivt_direction_values - mean_reference_deg + 180) % 360 - 180)
    pixel_count_in_range = (degrees_deviation <= range_degrees).sum()
    total_pixels = ivt_direction_values.size
    pct_in_range = round((pixel_count_in_range / total_pixels) * 100)
    is_coherent = pct_in_range >= 50
    # CP: keep below block for logging later on
    # if not is_coherent and pct_in_range > 0:
    #     print(f"{pct_in_range}% of pixels are within {range_degrees} of the mean IVT direction of {round(mean_reference_deg)}")
    return is_coherent * 1


def filter_regions_by_ivt_direction_coherence(regions, ivt_direction):
    """Filter by IVT directional coherence, e.g., objects with 50% of grid cells having deviation > 45° from the object's mean IVT will discarded."
    ## Note, while we are "in" the direction data we can probably hit this criteria too:
    Consistency Between Object Mean IVT Direction and Overall Orientation...
    """

    # prescribe CRS
    ivt_direction.rio.write_crs("epsg:4326", inplace=True)
    # reproject to 3338 with prescribed grid cell size to match xy dimensions of labeled data
    ivt_direction_3338 = ivt_direction.rio.reproject("epsg:3338", resolution=spatial_resolution_reprojected, resampling=rasterio.enums.Resampling.nearest)
    
    drop_dict = {}

    for region_arr, dir_arr in zip(regions, ivt_direction_3338):
        
        # get unique labels for each time stamp
        timestamp_region_labels = list(np.unique(region_arr.values))
        
        # check labeled region presence to avoid work on empty arrays
        # arrays with no regions should only have one unique value (0)
        if len(timestamp_region_labels) > 1:

            incoherent_labels_to_drop = []
            # use `labeled_comprehension` to get IVT directional coherence for each labeled region
            # index arg determines which labels will be used
            # this will determine whether to drop (0) or keep (1) each label for the time step
            coherence_results = labeled_comprehension(dir_arr, region_arr, index=timestamp_region_labels,
                                                     func=is_directionally_coherent, out_dtype=int, default=0)
            
            # first label is zero in this implementation, which we can skip
            incoherent_indices = [ix for ix, value in enumerate(coherence_results[1:]) if value == 0]    
            if len(incoherent_indices) > 0:
                # use the indices of where the coherence failure occurs to get which labels to drop
                for ix in incoherent_indices:
                    incoherent_labels_to_drop.append(timestamp_region_labels[1:][ix])
                drop_dict[region_arr.time.values] = incoherent_labels_to_drop
                
    # use the drop dictionary to loop through the original dataset and reassign dropped labels to 0
    for d in drop_dict:
        regions.loc[dict(time=d)] = xr.where(regions.sel(time=d).isin(drop_dict[d]), 0, regions.sel(time=d))
    return drop_dict

### pseudocode + notes, content below here not refactored

# there are 3 components to this following the rules of G&W 2015 (https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2015JD024257)

# 1. "Coherence in IVT Direction. If more than half of the grid cells have IVT deviating more than 45° from the object's mean IVT, the object is discarded."
# 2. "Object Mean Meridional IVT. ... an object is discarded if the mean IVT does not have an appreciable poleward component (>50 kg m−1 s−1)."
# this poleward component is the p72.162 variable in the ERA5 dataset, equivalent to northward "v" component of IVT
# using labeled comprehension, the code would look something like this:
# for r, p in zip(regions, ivt_poleward):
#     lbls = list(np.unique(r.values))
#     mean_poleward = labeled_comprehension(p, r, lbls, np.mean, float, 0)
# 3. "Consistency Between Object Mean IVT Direction and Overall Orientation...An object is discarded if the direction of mean IVT deviates from the overall orientation by more than 45°." G&W use the azimuth of the line drawn at maximum great circle distance of the labeled region. I think we would find the two cells within the labeled region that are at maximum distance from each other, find the line between them, and get its azimuth to compare to the mean direction calculated in step 1. NOTE that the degree directions used in step 1 represent direction the AR is coming FROM....so the azimuth here would also need to point in the FROM direction, not the TO direction....

#compute_ar_stats("AR", IVT_data_cube, "direction", "shape")
#compute general stats for all cells within final AR objects (eg min/mean/max IVT, direction, etc)
#compute basic non-stat values for final AR objects (eg length, width, area, date/time, unique AR id?)
#return long format "stats_df" dataframe with columns (AR / stat / value)
#ar_to_gdf("AR", "stats_df")
#convert all ARs to polygon geodataframe
#join AR stats as attributes
#(polygons will be overlapping in the case of long duration AR events)
#return geodataframe
