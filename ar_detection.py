"""This module will threshold the IVT Dataset and identify AR candidates. Each candidate will be tested to determine if it meets the various criteria for AR classification."""
import math
import xarray as xr
import numpy as np
import rasterio
import pyproj
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import label, generate_binary_structure, labeled_comprehension
from skimage.measure import regionprops
from scipy.stats import circmean
from haversine import haversine
from shapely.geometry import Polygon

from config import ar_params


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
    """Label contiguous geometric regions of IVT values that exceed the targent quantile and floor for each time step. Each region is labeled with a unique integer identifier. Output will have the same dimensions as the input IVT intensity mask.

    Parameters
    ----------
    ivt_mask : xarray.DataArray
        IVT magnitude where the values exceed the IVT quantile and IVT floor.

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
    
    # CP note: looping through the time dimension may be a chokepoint
    # CP note: perhaps a good candidate for vectorizing or parallelization
    for a in range(len(ivt_mask.time)):
        # label contiguous regions of each timestep
        labeled_array, num_features = label(ivt_mask[a].values, structure=s)
        # map labeled regions to the timestep in the template array
        da[a] = labeled_array
    return da


def generate_region_properties(labeled_blobs, ds):
    """
    Generate region properties and calculate criteria results for each labeled region in a time slice.

    Parameters
    ----------
    labeled_blobs : xarray.DataArray
        Labeled (with unique integers) discrete and contiguous regions of IVT threshold exceedance.
    ds : xarray.Dataset
        The original dataset containing variables: 'ivt_mag', 'ivt_dir', 'p72.162'.

    Returns
    -------
    dict
        A dictionary with region properties and calculated criteria results for each time slice.

    Notes
    -----
    The function processes each time slice of the labeled regions and calculates various properties and criteria
    results for each labeled region within that time slice. The output dictionary stores the region properties and
    criteria results for each labeled region at each time step.

    The 'labeled_blobs' input contains labeled regions with unique integer identifiers, where each region represents
    a contiguous area of IVT threshold exceedance.

    The 'ds' input is an xarray dataset that contains the following variables:
    - 'ivt_mag': IVT magnitude values for each time step.
    - 'ivt_dir': IVT direction values for each time step.
    - 'p72.162': IVT poleward values for each time step.

    The output dictionary 'ar_criteria_di' is organized as follows:
    - The keys represent timestamps (time values) of each time slice.
    - Each value is a sub-dictionary containing criteria results for each labeled region within that time slice.
    - The sub-dictionary for each labeled region contains measurements and criteria results.

    For each labeled region, the function calculates the following region properties:
    - 'blobs with IVT magnitude': Properties calculated using 'ivt_mag' as the intensity image.
    - 'blobs with IVT poleward': Properties calculated using 'ivt_poleward' as the intensity image.
    - 'blobs with IVT direction': Properties calculated using 'ivt_dir' as the intensity image.
    """
    ar_criteria_di = {}
    
    for labeled_timeslice, ivt_magnitude_arr, ivt_poleward_arr, ivt_dir_arr in zip(labeled_blobs,
                                                                                   ds["ivt_mag"],
                                                                                   ds["ivt_dir"],
                                                                                   ds["p72.162"]):
        timestamp = labeled_timeslice.time.values.astype(str)
        ar_criteria_di[timestamp] = {}
        
        # this sub-dict will store the measurments and criteria results
        ar_criteria_di[timestamp]["ar_candidate_int_labels"] = {}
        for i in np.unique(labeled_timeslice.astype(int).values)[1:]:
            ar_criteria_di[timestamp]["ar_candidate_int_labels"][i] = {}
    
        # generate lazy zonal statistics and shape measurements for each region within a time slice for all variables
        ar_criteria_di[timestamp]["blobs with IVT magnitude"] = regionprops(labeled_timeslice.astype(int).values,
                                                                                  intensity_image=ivt_magnitude_arr.data)
    
        ar_criteria_di[timestamp]["blobs with IVT poleward"] = regionprops(labeled_timeslice.astype(int).values,
                                                                                      intensity_image=ivt_poleward_arr.data)
        
        ar_criteria_di[timestamp]["blobs with IVT direction"] = regionprops(labeled_timeslice.astype(int).values,
                                                                                  intensity_image=ivt_dir_arr.data)
    return ar_criteria_di


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
        A tuple containing the label of the region and its corresponding length-to-width ratio.

    Notes
    -----
    The function calculates the length-to-width ratio of a labeled region using its major and minor axis lengths.
    The input 'blob' is a RegionProperties object from the skimage.measure.regionprops function. It represents
    a labeled region and contains various properties, including 'axis_major_length' and 'axis_minor_length'.

    The length-to-width ratio is calculated as the ratio of 'axis_major_length' to 'axis_minor_length'.
    If 'axis_minor_length' is zero or not available, the ratio is set to zero to avoid divide by zero errors.
    """
    try:
        length_width_ratio = (blob.axis_major_length / blob.axis_minor_length)
    except:
        # avoiding divide by zero errors
        length_width_ratio = 0
    return blob.label, round(length_width_ratio, 1)                                                          


def get_major_axis_haversine_distance(blob, ds):
    """
    Calculate the haversine distance of the major axis of a labeled region.

    Parameters
    ----------
    blob : skimage.measure._regionprops.RegionProperties
        Region properties object representing a labeled region.
    ds : xarray.Dataset
        The dataset containing latitude and longitude coordinates.

    Returns
    -------
    tuple
        A tuple containing the label of the region and the haversine distance of its major axis.

    Notes
    -----
    The function calculates the haversine distance of the major axis of a labeled region using the region's
    centroid and orientation properties. It also requires the input dataset ('ds') containing latitude and longitude
    coordinates for spatial reference.

    The major axis is defined by two points: the region's centroid and an endpoint that extends half the length
    of the major axis from the centroid along the region's orientation. The function calculates the latitude and
    longitude of these two points using the orientation and axis lengths provided by the 'blob' object.

    The coordinates of the centroid and major axis endpoint are then used to compute the haversine distance,
    which is the great-circle distance between these two points on the Earth's surface. The haversine distance
    is calculated using the 'haversine' function from the 'haversine' library, and the resulting distance is
    multiplied by 2 to get the total length of the major axis.

    The output is a tuple containing the label of the region and the haversine distance of its major axis.
    The distance is rounded to the nearest kilometer.
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
    
    # the total axis length will be twice the distance between these points
    # a haversine distance should be used because we expect points to be substantially far apart
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
        The dataset containing latitude and longitude coordinates.

    Returns
    -------
    tuple
        A tuple containing the label of the region and the forward azimuth of the bounding box diagonal.

    Notes
    -----
    The function calculates the forward azimuth of the bounding box diagonal of a labeled region. The bounding box
    diagonal is the line connecting the lower-left corner to the upper-right corner of the region's bounding box.

    The function extracts the latitude and longitude coordinates of the lower-left (minrow, mincol) and upper-right
    (maxrow, maxcol) corners of the bounding box. If any of the coordinates exceed the dimensions of the dataset
    ('ds'), they are corrected to ensure they fall within valid bounds.

    The forward azimuth of the bounding box diagonal is then computed using the 'pyproj.Geod' class from the 'pyproj'
    library. The 'geodesic.inv' method is used to calculate the azimuth, distance, and back azimuth between the
    lower-left and upper-right corners.

    The output is a tuple containing the label of the region and the forward azimuth of the bounding box diagonal.
    The azimuth is rounded to the nearest degree.
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
    fwd_azimuth, back_azimuth, distance = geodesic.inv(left_lon, lower_lat, right_lon, upper_lat)
    # we don't need the distance of the bounding box diagonal or the back azimuth
    return blob.label, round(fwd_azimuth)


def get_poleward_strength(blob):
    """
    Compute the poleward strength of a labeled region.

    Parameters
    ----------
    blob : skimage.measure._regionprops.RegionProperties
        Region properties object representing a labeled region.

    Returns
    -------
    tuple
        A tuple containing the label of the region and the poleward strength.

    Notes
    -----
    The function computes the poleward strength of a labeled region as the rounded mean intensity value of the region.
    The intensity value represents the strength of the poleward flow in the region.

    The output is a tuple containing the label of the region and the computed poleward strength rounded to the nearest
    integer value.
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
        A tuple containing the label of the region, the computed directional coherence percentage, and the mean
        reference degree.

    Notes
    -----
    The function calculates the directional coherence of a labeled region. The directional coherence represents the
    percentage of pixels in the region that have wind directions within a certain deviation threshold from the mean
    reference degree.

    The mean reference degree is calculated using the 'circmean' function from the 'scipy.stats' module. The 'circmean'
    function computes the circular mean of wind directions in degrees, considering the circular nature of the data.

    The degrees deviation is then calculated as the absolute difference between each pixel's wind direction and the mean
    reference degree, taking into account the circular nature of the data.

    The pixel count within the specified deviation threshold is determined, and the total number of pixels in the region
    is computed.

    The directional coherence is expressed as the percentage of coherent pixels within the deviation threshold over the
    total number of pixels in the region. The result is rounded to the nearest integer percentage.

    The output is a tuple containing the label of the region, the computed directional coherence percentage, and the
    mean reference degree.
    """
    mean_reference_deg = circmean(blob.image_intensity, high=360, low=0)
    degrees_deviation = abs((blob.image_intensity - mean_reference_deg + 180) % 360 - 180)
    pixel_count_in_range = (degrees_deviation <= ar_params["direction_deviation_threshold"]).sum()
    total_pixels = blob.image_intensity.size
    pct_coherent = round((pixel_count_in_range / total_pixels) * 100)
    return blob.label, pct_coherent, mean_reference_deg


def get_data_for_ar_criteria(ar_criteria_di, ds):
    """
    Calculate various AR candidate criteria for each time slice and labeled region.

    Parameters
    ----------
    ar_criteria_di : dict
        Dictionary containing region properties and measurements for AR candidates.
    ds : xarray.Dataset
        The dataset containing latitude and longitude coordinates.

    Returns
    -------
    dict
        Updated dictionary with additional AR candidate criteria.

    Notes
    -----
    The function calculates and adds various AR candidate criteria to the input dictionary 'ar_criteria_di'. The AR
    candidate criteria include length/width ratio, major axis length in kilometers, overall orientation (azimuth), mean
    poleward strength, and directional coherence.

    The function iterates through each time slice in the dictionary 'ar_criteria_di' and processes the labeled regions
    within each time slice. For each region, the function calls specific helper functions to compute the criteria.

    The computed AR candidate criteria are stored in the dictionary 'ar_criteria_di' for each labeled region and time
    slice under appropriate keys.

    The updated dictionary with the additional AR candidate criteria is returned.
    """
    for k in tqdm(ar_criteria_di, desc="Getting length/width ratio for each AR candidate for each timeslice..."):
        for blob in ar_criteria_di[k]["blobs with IVT magnitude"]:
            int_label, ratio = get_length_width_ratio(blob)
            ar_criteria_di[k]["ar_candidate_int_labels"][int_label]["length/width ratio"] = ratio
    
    for k in tqdm(ar_criteria_di, desc="Getting axis length (km) for each AR candidate for each timeslice..."):
        for blob in ar_criteria_di[k]["blobs with IVT magnitude"]:
            int_label, major_axis_length_km = get_major_axis_haversine_distance(blob, ds)
            ar_criteria_di[k]["ar_candidate_int_labels"][int_label]["major axis length (km)"] = major_axis_length_km
    
    for k in tqdm(ar_criteria_di, desc="Getting overall orientation (azimuth) for each AR candidate for each timeslice..."):
        for blob in ar_criteria_di[k]["blobs with IVT magnitude"]:
            int_label, max_distance_mean_azimuth = get_azimuth_of_furthest_points(blob, ds)
            ar_criteria_di[k]["ar_candidate_int_labels"][int_label]["overall orientation"] = max_distance_mean_azimuth

    for k in tqdm(ar_criteria_di, desc="Getting mean poleward strength for each AR candidate for each timeslice..."):
        for blob in ar_criteria_di[k]["blobs with IVT poleward"]:
            int_label, mean_poleward_strength = get_poleward_strength(blob)
            ar_criteria_di[k]["ar_candidate_int_labels"][int_label]["mean poleward strength"] = mean_poleward_strength

    for k in tqdm(ar_criteria_di, desc="Getting directional coherence for each AR candidate for each timeslice..."):
        for blob in ar_criteria_di[k]["blobs with IVT direction"]:
            int_label, pct_coherent, mean_of_ivt_dir = get_directional_coherence(blob)
            ar_criteria_di[k]["ar_candidate_int_labels"][int_label]["directional_coherence"] = pct_coherent
            ar_criteria_di[k]["ar_candidate_int_labels"][int_label]["mean_of_ivt_dir"] = round(mean_of_ivt_dir)
    
    return ar_criteria_di


def apply_criteria(ar_criteria_di):
    """
    Apply criteria to determine AR candidates based on calculated measurements.

    Parameters
    ----------
    ar_criteria_di : dict
        Dictionary containing region properties and measurements for AR candidates.

    Returns
    -------
    dict
        Updated dictionary with AR candidates' classification based on criteria.

    Notes
    -----
    The function applies specific criteria to determine whether a labeled region qualifies as an AR candidate. The criteria
    include coherence in IVT direction, object mean meridional IVT, consistency between object mean IVT direction and overall
    orientation, length, and length/width ratio.

    The function iterates through each time slice in the dictionary 'ar_criteria_di' and processes the labeled regions within
    each time slice. For each labeled region, the function checks the calculated measurements against specific thresholds to
    determine if the region meets the criteria.

    The AR candidate classification results (True or False) are stored in the dictionary 'ar_criteria_di' for each labeled
    region and time slice under appropriate keys.

    The updated dictionary with the AR candidates' classification is returned.
    """
    criteria = ["Coherence in IVT Direction", "Object Mean Meridional IVT",
                    "Consistency Between Object Mean IVT Direction and Overall Orientation", "Length" ,"Length/Width Ratio"]
    
    for k in tqdm(ar_criteria_di):
        for blob_label in ar_criteria_di[k]["ar_candidate_int_labels"]:
    
            for criterion in criteria:
                ar_criteria_di[k]["ar_candidate_int_labels"][blob_label][criterion] = False
            
            if ar_criteria_di[k]["ar_candidate_int_labels"][blob_label]["length/width ratio"] > 2:
                ar_criteria_di[k]["ar_candidate_int_labels"][blob_label]["Length/Width Ratio"] = True
        
            if ar_criteria_di[k]["ar_candidate_int_labels"][blob_label]["major axis length (km)"] > ar_params["min_axis_length"]:
                    ar_criteria_di[k]["ar_candidate_int_labels"][blob_label]["Length"] = True

            # Determine IVT directional coherence. If more than half of the grid cells have IVT directions deviating
            # greater than the threshold from the object's mean IVT, then then the object is not directionally coherent.
            if ar_criteria_di[k]["ar_candidate_int_labels"][blob_label]["directional_coherence"] > 50:
                    ar_criteria_di[k]["ar_candidate_int_labels"][blob_label]["Coherence in IVT Direction"] = True
            
            # Determine strength of IVT poleward component. If the object's mean northward IVT component is less than the criterion, 
            # then the object lacks a strong poleward component and should be rejected from the AR classification.
            if ar_criteria_di[k]["ar_candidate_int_labels"][blob_label]["mean poleward strength"] > ar_params["mean_meridional"]:
                ar_criteria_di[k]["ar_candidate_int_labels"][blob_label]["Object Mean Meridional IVT"] = True
            
            # Determine Consistency Between Object Mean IVT Direction and Overall Orientation.
            # if the direction of mean IVT deviates from the overall orientation by more than 45°, then reject from AR classification
            if abs(180 - (180 - ar_criteria_di[k]["ar_candidate_int_labels"][blob_label]["mean_of_ivt_dir"] + ar_criteria_di[k]["ar_candidate_int_labels"][blob_label]["overall orientation"]) % 360) < ar_params["orientation_deviation_threshold"]:
                ar_criteria_di[k]["ar_candidate_int_labels"][blob_label]["Consistency Between Object Mean IVT Direction and Overall Orientation"] = True
    
            # how many criteria were met?
            ar_criteria_di[k]["ar_candidate_int_labels"][blob_label]["Criteria Passed"] = list(ar_criteria_di[k]["ar_candidate_int_labels"][blob_label].values()).count(True)

    return ar_criteria_di


def filter_ars(ar_criteria_di, n_criteria_required=5):
    """Produce dict with timestamps and region IDs for ARs meeting criteria.
    
    Filter AR candidates that meet the specified number of criteria.

    Parameters
    ----------
    ar_criteria_di : dict
        Dictionary containing region properties and measurements for AR candidates.
    n_criteria_required : int, optional
        Number of criteria that a region must meet to be considered an AR candidate. Default is 5.

    Returns
    -------
    dict
        Dictionary with timestamps and region IDs for ARs that meet the specified number of criteria.

    Notes
    -----
    The function takes the dictionary 'ar_criteria_di', which contains region properties and measurements for AR candidates,
    and filters the candidates based on the specified number of criteria they must meet to be considered ARs.

    The function iterates through each time slice in 'ar_criteria_di' and checks each labeled region within that time slice.
    If a region has met the required number of criteria (determined by 'n_criteria_required'), it is added to the 'filtered_ars'
    dictionary along with the timestamp and region ID.

    The 'filtered_ars' dictionary contains timestamps as keys and region IDs as values for the ARs that meet the specified
    number of criteria.

    The updated dictionary 'filtered_ars' is returned.
    """
    filtered_ars = {}
    for k in tqdm(ar_criteria_di):
        for blob_label in ar_criteria_di[k]["ar_candidate_int_labels"]:
            if ar_criteria_di[k]["ar_candidate_int_labels"][blob_label]["Criteria Passed"] >= n_criteria_required:
                filtered_ars[k] = blob_label
    return filtered_ars


def binary_to_geodataframe(binary_data):
    """
    Convert a binary 2D array to a GeoDataFrame with polygons representing the regions of value 1.

    Parameters
    ----------
    binary_data : numpy.ndarray
        Binary 2D array containing values of 0 and 1.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with polygons representing the regions of value 1 in the binary array.

    Notes
    -----
    The function takes a binary 2D array 'binary_data' and converts it into a GeoDataFrame with polygons. Each value of 1 in the
    binary array corresponds to a polygon in the GeoDataFrame.

    The function iterates through each cell in the binary array and checks if the value is equal to 1. If the value is 1, it
    creates a polygon with coordinates representing the cell's location in the binary array.

    The polygons are collected in a list, and a GeoDataFrame is created using the list of polygons, where each polygon
    represents a region of value 1 in the binary array. The GeoDataFrame is assigned the CRS (Coordinate Reference System)
    EPSG:4326.

    The resulting GeoDataFrame is returned.
    """
    shapes = []
    for i in range(binary_data.shape[0] - 1):  
        for j in range(binary_data.shape[1] - 1):
            if binary_data[i, j] == 1:
                # Define the geometry for each binary value (cell with value 1)
                lon_min, lat_min = binary_data.longitude[j], binary_data.latitude[i]
                lon_max, lat_max = binary_data.longitude[j + 1], binary_data.latitude[i + 1]
                # make many 1x1 polygons
                poly = Polygon([(lon_min, lat_min), (lon_min, lat_max), (lon_max, lat_max), (lon_max, lat_min)])
                shapes.append(poly)

    gdf = gpd.GeoDataFrame({"_value": [1] * len(shapes)}, geometry=shapes, crs=f"EPSG:4326")
    return gdf


def create_geodataframe_with_all_ars(filtered_ars, ar_criteria_di, labeled_blobs):
    """
    Create a GeoDataFrame containing all ARs (Atmospheric Rivers) meeting the criteria.

    Parameters
    ----------
    filtered_ars : dict
        Dictionary with timestamps and region IDs for ARs meeting the criteria.
    ar_criteria_di : dict
        Dictionary containing AR criteria information for each timestamp.
    labeled_blobs : xarray.DataArray
        Labeled discrete and contiguous regions of IVT threshold exceedance.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing polygons representing all ARs meeting the criteria.

    Notes
    -----
    The function takes a dictionary 'filtered_ars' containing timestamps and region IDs of ARs that meet the criteria. It also
    takes 'ar_criteria_di', a dictionary containing AR criteria information for each timestamp, and 'labeled_blobs', a
    DataArray representing labeled discrete and contiguous regions of IVT threshold exceedance.

    The function iterates through each AR in 'filtered_ars' and retrieves the corresponding region from 'labeled_blobs'.
    It then converts the region into a binary mask using 'binary_to_geodataframe' function and creates a GeoDataFrame with
    polygons representing each AR.

    The attributes of each AR, such as mean IVT, max IVT, min IVT, and timestamp, are added as columns to the GeoDataFrame.
    The GeoDataFrame is returned, containing all ARs that meet the criteria.
    """
    single_ar_gdfs = []
    
    for k in tqdm(filtered_ars):
        blob_ix = filtered_ars[k] - 1
        ar_blob = ar_criteria_di[k]["blobs with IVT magnitude"][blob_ix]
        ar_mask = (labeled_blobs.sel(time=k) == filtered_ars[k])
        ar_mask.rio.write_crs("epsg:4326", inplace=True)
        gdf = binary_to_geodataframe(ar_mask)
        # dummy column for dissolving all the 1x1 polygons into a single polygon
        gdf["_column"] = 0
        ar_gdf = gdf.dissolve(by="_column")
        # add some attributes, can add more here, e.g., mean transport direction
        ar_gdf["mean IVT"] = round(ar_criteria_di[k]["blobs with IVT magnitude"][blob_ix].intensity_mean)
        ar_gdf["max IVT"] = round(ar_criteria_di[k]["blobs with IVT magnitude"][blob_ix].intensity_max)
        ar_gdf["min IVT"] = round(ar_criteria_di[k]["blobs with IVT magnitude"][blob_ix].intensity_min)
        ar_gdf["time"] = k
        del ar_gdf["_value"]
    
        single_ar_gdfs.append(ar_gdf)
    
    all_ars = pd.concat(single_ar_gdfs)
    return all_ars


def create_shapefile(all_ars):
    """
    Create a shapefile from the GeoDataFrame containing all ARs meeting the criteria.

    Parameters
    ----------
    all_ars : geopandas.GeoDataFrame
        GeoDataFrame containing polygons representing all ARs meeting the criteria.

    Returns
    -------
    None

    Notes
    -----
    The function takes a GeoDataFrame 'all_ars' containing polygons representing all ARs that meet the criteria.
    It then saves the GeoDataFrame as a shapefile named 'ar_output.shp' in the current working directory.
    """
    all_ars.to_file("ar_output.shp")
