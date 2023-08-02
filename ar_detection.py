"""This module will threshold the IVT Dataset and identify AR candidates. Each candidate is tested to determine if it meets the various criteria for AR classification."""
import math

import xarray as xr
import numpy as np
import pyproj
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import label, generate_binary_structure
from skimage.measure import regionprops
from scipy.stats import circmean
from haversine import haversine
from shapely.geometry import Polygon

from config import ar_params, ard_fp, shp_fp


def compute_intensity_mask(ivt_mag, ivt_quantile, ivt_floor):
    """Compute IVT mask where IVT magnitude exceeds quantile and floor values.

    Parameters
    ----------
    ivt_mag : xarray.DataArray
        IVT magnitude
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
    """Label contiguous geometric regions of IVT values that exceed the target quantile and floor for each time step. Each region is labeled with a unique integer identifier. Output will have the same dimensions as the input IVT intensity mask.

    Parameters
    ----------
    ivt_mask : xarray.DataArray
        IVT magnitude where the values exceed the IVT quantile and IVT floor.

    Returns
    -------
    xarray.DataArray
        Labeled (with unique integers) contiguous regions of IVT threshold exceedance.
    """
    # set search structure to include diagonal neighbors
    s = generate_binary_structure(2,2)
    # initialize output by copying ivt_mask, renaming it, setting values to nodata value
    # this constructs a template DataArray with the same size/dimensions as the input to store region labeling results
    da = ivt_mask.copy(deep=False).rename('regions').where(ivt_mask.values == -9999)
    
    # CP note: perhaps a good candidate for vectorizing or parallelization
    for a in range(len(ivt_mask.time)):
        # label contiguous regions of each timestep
        labeled_array, num_features = label(ivt_mask[a].values, structure=s)
        # map labeled regions to the timestep in the template array
        da[a] = labeled_array
    return da


def generate_region_properties(labeled_blobs, ds):
    """
    Generate region properties for all variables for each region in a time slice.

    Parameters
    ----------
    labeled_blobs : xarray.DataArray
        Labeled (with unique integers) contiguous regions of IVT threshold exceedance.
    ds : xarray.Dataset
        The original dataset containing variables: 'ivt_mag', 'ivt_dir', 'p72.162'.

    Returns
    -------
    dict
        region properties with criteria data results for each time slice.

    Notes
    -----
    Processes each time slice of the labeled regions and calculates various properties.
    The 'ds' input contains the following variables at each time step and they are used as the intensity image input for the computed region properties:
    - 'ivt_mag': IVT magnitude values
    - 'ivt_dir': IVT direction values
    - 'p72.162': IVT poleward component values
    """
    ar_di = {}
    for labeled_slice, ivt_magnitude, ivt_poleward, ivt_dir in zip(labeled_blobs,
                                                                   ds["ivt_mag"],
                                                                   ds["ivt_dir"],
                                                                   ds["p72.162"]):
        timestamp = labeled_slice.time.values.astype(str)
        ar_di[timestamp] = {}
        
        # this sub-dict will store the measurements and criteria results
        ar_di[timestamp]["ar_targets"] = {}
        for i in np.unique(labeled_slice.astype(int).values)[1:]:
            ar_di[timestamp]["ar_targets"][i] = {}
    
        # generate lazy zonal statistics, shape metrics for each region within a time slice for all variables
        ar_di[timestamp]["blobs with IVT magnitude"] = regionprops(labeled_slice.astype(int).values, intensity_image=ivt_magnitude.data)
    
        ar_di[timestamp]["blobs with IVT poleward"] = regionprops(labeled_slice.astype(int).values, intensity_image=ivt_poleward.data)
        
        ar_di[timestamp]["blobs with IVT direction"] = regionprops(labeled_slice.astype(int).values,intensity_image=ivt_dir.data)
    return ar_di


def get_length_width_ratio(blob):
    """
    Calculate the length-to-width ratio for a labeled region.

    Parameters
    ----------
    blob : skimage.measure._regionprops.RegionProperties
        Region properties object representing a labeled region.

    Returns
    -------
    tuple
        (label of the region, corresponding length/width ratio)

    Notes
    -----
    The function calculates the length-to-width ratio of a labeled region using its major and minor axis lengths. The input 'blob' is a RegionProperties object from the skimage.measure.regionprops function. It represents a labeled region and contains various properties, including 'axis_major_length' and 'axis_minor_length'.
    The length-to-width ratio is calculated as the ratio of 'axis_major_length' to 'axis_minor_length'.
    """
    try:
        length_width_ratio = (blob.axis_major_length / blob.axis_minor_length)
    except:
        # for divide by zero errors
        length_width_ratio = 0
    length_width_ratio = round(length_width_ratio, 1)
    return blob.label, length_width_ratio
                                                      

def get_major_axis_haversine_distance(blob, ds):
    """
    Calculate the haversine distance of the major axis of a labeled region.

    Parameters
    ----------
    blob : skimage.measure._regionprops.RegionProperties
        Region properties object representing a labeled region.
    ds : xarray.Dataset
        must have latitude and longitude coordinates.

    Returns
    -------
    tuple
        (label of the region, haversine length (km) of its major axis)

    Notes
    -----
    The function calculates the haversine distance of the major axis of a labeled region using the region's centroid and orientation properties. It also requires the input dataset ('ds') containing latitude and longitude
    coordinates for spatial reference.

    The major axis is defined by two points: the region's centroid and an endpoint that extends half the length of the major axis from the centroid along the region's orientation. The function calculates the latitude and longitude of these two points using the orientation and axis lengths provided by the 'blob' object.

    The coordinates of the centroid and major axis endpoint are then used to compute the haversine distance, which is the great-circle distance between these two points on the Earth's surface. The haversine distance is calculated using the 'haversine' function from the 'haversine' library, and the resulting distance is multiplied by 2 to get the total length of the major axis.
    """
    y0, x0 = blob.centroid
    orientation = blob.orientation
    # find endpoints of the minor and major axes with respect to the centroid
    # note that we force integers because we ultimately desire array indices
    y0 = int(y0)
    x0 = int(x0)
    x1 = int(x0 + math.cos(orientation) * 0.5 * blob.axis_minor_length)
    y1 = int(y0 - math.sin(orientation) * 0.5 * blob.axis_minor_length)
    x2 = int(x0 - math.sin(orientation) * 0.5 * blob.axis_major_length)
    y2 = int(y0 - math.cos(orientation) * 0.5 * blob.axis_major_length)

    # the ellipse model used to find the orientation and major/minor axes lengths may extend into cartesian space that is beyond
    # the geographic extent of the data, so we will check and correct
    if x1 < 0:
        x1 = 0
    elif x1 >= ds.longitude.shape[0]:
        x1 = ds.longitude.shape[0] - 1
    if x2 < 0:
        x2 = 0
    elif x2 >= ds.longitude.shape[0]:
        x2 = ds.longitude.shape[0] - 1
    if y1 < 0:
        y1 = 0
    elif y1 >= ds.latitude.shape[0]:
        y1 = ds.latitude.shape[0] - 1
    if y2 < 0:
        y2 = 0
    elif y2 >= ds.latitude.shape[0]:
        y2 = ds.latitude.shape[0] - 1

    # use the array indices to select the latitude and longitude of the centroid and a major axis endpoint
    centroid_lat = ds.latitude[y0].values
    centroid_lon = ds.longitude[x0].values
    geo_centroid = (centroid_lat, centroid_lon)
    major_lat = ds.latitude[y2].values
    major_lon = ds.longitude[x2].values
    geo_major_axis_endpoint = (major_lat, major_lon)
    
    # total axis length will be twice the distance between these points
    # haversine distance used because we expect points to be substantially far apart
    half_major_axis_length = haversine(geo_centroid, geo_major_axis_endpoint)
    # km is default unit for haversine function
    major_axis_length_km = round(half_major_axis_length * 2)
    return blob.label, major_axis_length_km


def get_azimuth_of_furthest_points(blob, ds):
    """
    Compute the forward azimuth of the bounding box diagonal of a labeled region.

    Parameters
    ----------
    blob : skimage.measure._regionprops.RegionProperties
        Region properties object representing a labeled region.
    ds : xarray.Dataset
        must have latitude and longitude coordinates.

    Returns
    -------
    tuple
        (label of the region, forward azimuth of the bounding box diagonal)

    Notes
    -----
    The bounding box diagonal is the line connecting the lower-left corner to the upper-right corner of the region's bounding box. This is making an assumption, but given the configuration of land and ocean in the NW Pacific region, it is likely appropriate. If any bounding box coordinates exceed the dimensions of the dataset because of indexing, they are corrected to ensure they fall within valid bounds. The 'geodesic.inv' method from the pyproj library calculates the forward azimuth between the lower-left and upper-right corners.
    """
    minrow, mincol, maxrow, maxcol = blob.bbox

    if maxrow >= ds.latitude.shape[0]:
        maxrow = ds.latitude.shape[0] - 1
    if maxcol >= ds.longitude.shape[0]:
        maxcol = ds.longitude.shape[0] - 1
    
    lower_lat = ds.latitude[minrow].values
    left_lon = ds.longitude[mincol].values
    upper_lat = ds.latitude[maxrow].values
    right_lon = ds.longitude[maxcol].values
    
    geodesic = pyproj.Geod(ellps="WGS84")
    fwd_azimuth, _back_azimuth, _distance = geodesic.inv(left_lon, lower_lat, right_lon, upper_lat)
    
    return blob.label, round(fwd_azimuth)


def get_poleward_strength(blob):
    """
    Computes the poleward strength of a labeled region as the rounded mean intensity (poleward flow) value of the region.

    Parameters
    ----------
    blob : skimage.measure._regionprops.RegionProperties
        Region properties object representing a labeled region.

    Returns
    -------
    tuple
        (label of the region, mean poleward strength)
    """
    return blob.label, round(blob.intensity_mean)    


def get_directional_coherence(blob):
    """
    Compute the directional coherence of a labeled region.

    Parameters
    ----------
    blob : skimage.measure._regionprops.RegionProperties
        Region properties object representing a labeled region.

    Returns
    -------
    tuple
        (the label of the region, directional coherence percentage, mean IVT direction)

    Notes
    -----
    Directional coherence represents the percentage of pixels in the region that have wind directions within a certain deviation threshold from the mean IVT direction. Mean IVT direction computed the 'circmean' function from the 'scipy.stats' module to account for the polar coordinates of the direction variable. Result is rounded to the nearest integer percentage.
    """
    mean_reference_deg = circmean(blob.image_intensity, high=360, low=0)
    deg_deviation = abs((blob.image_intensity - mean_reference_deg + 180) % 360 - 180)
    pixel_count = (deg_deviation <= ar_params["direction_deviation_threshold"]).sum()
    total_pixels = blob.image_intensity.size
    pct_coherent = round((pixel_count / total_pixels) * 100)
    return blob.label, pct_coherent, mean_reference_deg


def get_data_for_ar_criteria(ar_di, ds):
    """
    Calculate various AR candidate criteria for each time slice and labeled region from the measured region properties.

    Parameters
    ----------
    ar_di : dict
        Dictionary containing region properties and measurements for AR candidates.
    ds : xarray.Dataset
        must have latitude and longitude coordinates.

    Returns
    -------
    dict
        Updated dictionary with additional AR candidate criteria.

    Notes
    -----
    Calls specific helper functions to compute the data needed for testing AR candidate criteria.
    """
    for k in tqdm(ar_di, desc="Getting length/width ratio for each AR target:"):
        for blob in ar_di[k]["blobs with IVT magnitude"]:
            int_label, ratio = get_length_width_ratio(blob)
            ar_di[k]["ar_targets"][int_label]["length/width ratio"] = ratio
    
    for k in tqdm(ar_di, desc="Getting axis length (km) for each AR target:"):
        for blob in ar_di[k]["blobs with IVT magnitude"]:
            int_label, major_axis_length_km = get_major_axis_haversine_distance(blob, ds)
            ar_di[k]["ar_targets"][int_label]["major axis length (km)"] = major_axis_length_km
    
    for k in tqdm(ar_di, desc="Getting overall orientation (azimuth) for each AR target:"):
        for blob in ar_di[k]["blobs with IVT magnitude"]:
            int_label, max_distance_mean_azimuth = get_azimuth_of_furthest_points(blob, ds)
            ar_di[k]["ar_targets"][int_label]["overall orientation"] = max_distance_mean_azimuth

    for k in tqdm(ar_di, desc="Getting mean poleward strength for each AR target:"):
        for blob in ar_di[k]["blobs with IVT poleward"]:
            int_label, mean_poleward_strength = get_poleward_strength(blob)
            ar_di[k]["ar_targets"][int_label]["mean poleward strength"] = mean_poleward_strength

    for k in tqdm(ar_di, desc="Getting directional coherence for each AR target:"):
        for blob in ar_di[k]["blobs with IVT direction"]:
            int_label, pct_coherent, mean_of_ivt_dir = get_directional_coherence(blob)
            ar_di[k]["ar_targets"][int_label]["directional_coherence"] = pct_coherent
            ar_di[k]["ar_targets"][int_label]["mean_of_ivt_dir"] = round(mean_of_ivt_dir)
    
    return ar_di


def apply_criteria(ar_di):
    """
    Apply criteria to determine AR candidates based on calculated measurements.

    Parameters
    ----------
    ar_di : dict
        Dictionary containing region properties and measurements for AR candidates.

    Returns
    -------
    dict
        Updated dictionary with AR candidates' classification based on criteria.

    Notes
    -----
    Criteria include coherence in IVT direction, object mean meridional IVT, consistency between object mean IVT direction and overall orientation, length, and length/width ratio. Function iterates through each time slice and processes the labeled regions within each time slice. For each labeled region, the function checks the calculated measurements against specific thresholds to determine if the region meets the criteria. The AR candidate classification results (True or False) are stored under appropriate keys.
    """
    criteria = ["Coherence in IVT Direction", "Mean Meridional IVT",
                    "Consistency Between Mean IVT Direction and Overall Orientation", "Length" ,"Length/Width Ratio"]
    
    for k in tqdm(ar_di):
        for blob_label in ar_di[k]["ar_targets"]:
    
            for criterion in criteria:
                ar_di[k]["ar_targets"][blob_label][criterion] = False
            
            if ar_di[k]["ar_targets"][blob_label]["length/width ratio"] > 2:
                ar_di[k]["ar_targets"][blob_label]["Length/Width Ratio"] = True
            # Axis length criterion
            if ar_di[k]["ar_targets"][blob_label]["major axis length (km)"] > ar_params["min_axis_length"]:
                    ar_di[k]["ar_targets"][blob_label]["Length"] = True
            # Directional coherence criterion
            if ar_di[k]["ar_targets"][blob_label]["directional_coherence"] > 50:
                ar_di[k]["ar_targets"][blob_label]["Coherence in IVT Direction"] = True
            # Strong poleward component criterion
            if ar_di[k]["ar_targets"][blob_label]["mean poleward strength"] > ar_params["mean_meridional"]:
                ar_di[k]["ar_targets"][blob_label]["Mean Meridional IVT"] = True
            
            # Consistency Between Mean IVT Direction and Overall Orientation criterion
            if abs(180 - (180 - ar_di[k]["ar_targets"][blob_label]["mean_of_ivt_dir"] + ar_di[k]["ar_targets"][blob_label]["overall orientation"]) % 360) < ar_params["orientation_deviation_threshold"]:
                ar_di[k]["ar_targets"][blob_label]["Consistency Between Mean IVT Direction and Overall Orientation"] = True
    
            # how many criteria were met?
            ar_di[k]["ar_targets"][blob_label]["Criteria Passed"] = list(ar_di[k]["ar_targets"][blob_label].values()).count(True)
    return ar_di


def filter_ars(ar_di, n_criteria_required=5):
    """
    Filter AR candidates that meet the specified number of criteria.

    Parameters
    ----------
    ar_di : dict
        Dictionary containing region properties and measurements for AR candidates.
    n_criteria_required : int, optional
        Number criteria (default=5) a region must meet to be considered an AR.

    Returns
    -------
    dict
        contains timestamps as keys and region IDs as values for the ARs that meet the specified number of criteria

    Notes
    -----
    The function iterates through each time slice in 'ar_di' and checks each labeled region within that time slice. If a region has met the required number of criteria (determined by 'n_criteria_required'), it is added to the 'filtered_ars' dictionary along with the timestamp and region ID.
    """
    filtered_ars = {}
    for k in tqdm(ar_di):
        for blob_label in ar_di[k]["ar_targets"]:
            if ar_di[k]["ar_targets"][blob_label]["Criteria Passed"] >= n_criteria_required:
                filtered_ars[k] = blob_label
    return filtered_ars


def binary_to_geodataframe(binary_data):
    """
    Convert a binary 2D array to a GeoDataFrame with polygons.

    Parameters
    ----------
    binary_data : numpy.ndarray
        Binary 2D array containing values of 0 and 1.

    Returns
    -------
    geopandas.GeoDataFrame
        EPSG:4326 polygons representing True / 1 regions in the binary array.

    Notes
    -----
    Each True value in the input corresponds to a polygon in the GeoDataFrame.
    """
    shapes = []
    # will make many 1 cell x 1 cell polygons
    for i in range(binary_data.shape[0] - 1):  
        for j in range(binary_data.shape[1] - 1):
            if binary_data[i, j] == 1:
                # Define the geometry for each binary value (cell with value 1)
                lon_min, lat_min = binary_data.longitude[j], binary_data.latitude[i]
                lon_max, lat_max = binary_data.longitude[j + 1], binary_data.latitude[i + 1]
                poly = Polygon([(lon_min, lat_min), (lon_min, lat_max), (lon_max, lat_max), (lon_max, lat_min)])
                shapes.append(poly)

    gdf = gpd.GeoDataFrame({"_value": [1] * len(shapes)}, geometry=shapes, crs=f"EPSG:4326")
    return gdf


def create_geodataframe_with_all_ars(filtered_ars, ar_di, labeled_blobs):
    """
    Create a GeoDataFrame containing all ARs meeting the criteria.

    Parameters
    ----------
    filtered_ars : dict
        Dictionary with timestamps and region IDs for ARs meeting the criteria.
    ar_di : dict
        Dictionary containing AR criteria information for each timestamp.
    labeled_blobs : xarray.DataArray
        Labeled discrete and contiguous regions of IVT threshold exceedance.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing polygons representing all ARs meeting the criteria.

    Notes
    -----
    Attributes of each AR, such as mean IVT, max IVT, min IVT, and time, are added as columns to the GeoDataFrame.
    """
    single_ar_gdfs = []
    
    for k in tqdm(filtered_ars):
        blob_ix = filtered_ars[k] - 1
        
        ar_mask = (labeled_blobs.sel(time=k) == filtered_ars[k])
        ar_mask.rio.write_crs("epsg:4326", inplace=True)
        gdf = binary_to_geodataframe(ar_mask)
        # dummy column for dissolving all the 1x1 polygons into a single polygon
        gdf["_column"] = 0
        ar_gdf = gdf.dissolve(by="_column")
        # add some attributes, can add more here, e.g., mean transport direction
        ar_gdf["mean IVT"] = round(ar_di[k]["blobs with IVT magnitude"][blob_ix].intensity_mean)
        ar_gdf["max IVT"] = round(ar_di[k]["blobs with IVT magnitude"][blob_ix].intensity_max)
        ar_gdf["min IVT"] = round(ar_di[k]["blobs with IVT magnitude"][blob_ix].intensity_min)
        ar_gdf["time"] = k
        del ar_gdf["_value"]
    
        single_ar_gdfs.append(ar_gdf)
    
    all_ars = pd.concat(single_ar_gdfs)
    return all_ars


def create_shapefile(all_ars, fp):
    """
    Save a shapefile to disk from the GeoDataFrame containing all ARs meeting the criteria.

    Parameters
    ----------
    all_ars : geopandas.GeoDataFrame
        GeoDataFrame containing polygons representing all ARs meeting the criteria.

    Returns
    -------
    None
    """
    all_ars.to_file(fp)


def detect_all_ars(fp, n_criteria, out_shp):
    """Run the entire AR detection pipeline and generate shapefile output.

    Parameters
    ----------
    fp : str
        File path to the IVT dataset in NetCDF format.
    n_criteria : int
        The number of criteria required to consider a region an AR.
    out_shp : str
        File path to save the shapefile output.

    Returns
    -------
    None
    """
    with xr.open_dataset(fp) as ivt_ds:
        ivt_ds.rio.write_crs("epsg:4326", inplace=True)
        ivt_ds["thresholded"] = compute_intensity_mask(ivt_ds["ivt_mag"], ivt_ds["ivt_quantile"], ar_params["ivt_floor"])
        labeled_regions = label_contiguous_mask_regions(ivt_ds["thresholded"])
        ar_di = generate_region_properties(labeled_regions, ivt_ds)
        ar_di = get_data_for_ar_criteria(ar_di, ivt_ds)
        ar_di = apply_criteria(ar_di)
        output_ars = filter_ars(ar_di, n_criteria_required=n_criteria)
        output_ar_gdf = create_geodataframe_with_all_ars(output_ars, ar_di, labeled_regions)
        create_shapefile(output_ar_gdf, out_shp)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect atmospheric rivers and generate shapefile output.")
    parser.add_argument("--input_file", type=str, default=ard_fp, help="File path to the IVT dataset in NetCDF format.")
    parser.add_argument("--n_criteria", type=int, default=5, help="Number criteria required to consider a region an AR.")
    parser.add_argument("--output_shapefile", type=str, default=shp_fp, help="File path to save the shapefile output.")

    args = parser.parse_args()

    print(f"Using the IVT input {args.input_file} to filter candidate ARs based on {n_criteria} criteria.")
    
    detect_all_ars(fp=args.input_file, n_criteria=args.n_criteria, out_shp=args.output_shapefile)
    
    print(f"Processing complete, shapefile output written to {args.output_shapefile}.")
    
