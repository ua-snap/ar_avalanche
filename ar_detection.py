"""This module will threshold the IVT Dataset and identify AR candidates. Each candidate is tested to determine if it meets the various criteria for AR classification. Candidate areas are converted to polygons and exported as a shapefile."""
import math

import xarray as xr
import numpy as np
import pyproj
import geopandas as gpd
import pandas as pd
import shapely
from tqdm import tqdm
from scipy.ndimage import label, generate_binary_structure
from skimage.measure import regionprops
from scipy.stats import circmean
from scipy.sparse.csgraph import connected_components
from haversine import haversine
from shapely.geometry import shape
from rasterio.features import shapes
from datetime import datetime


from config import (
    ar_params,
    ard_fp,
    shp_fp,
    csv_fp,
    ak_shp,
    landfall_shp,
    landfall_csv,
    landfall_events_shp,
    landfall_events_csv
)


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
    return xr.apply_ufunc(func, ivt_mag, ivt_quantile, ivt_floor, dask="parallelized")


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
    s = generate_binary_structure(2, 2)
    # initialize output by copying ivt_mask, renaming it, setting values to nodata value
    # this constructs a template DataArray with the same size/dimensions as the input to store region labeling results
    da = ivt_mask.copy(deep=False).rename("regions").where(ivt_mask.values == -9999)

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
    for labeled_slice, ivt_magnitude, ivt_poleward, ivt_dir in zip(
        labeled_blobs, ds["ivt_mag"], ds["ivt_dir"], ds["p72.162"]
    ):
        timestamp = labeled_slice.time.values.astype(str)
        ar_di[timestamp] = {}

        # this sub-dict will store the measurements and criteria results
        ar_di[timestamp]["ar_targets"] = {}
        for i in np.unique(labeled_slice.astype(int).values)[1:]:
            ar_di[timestamp]["ar_targets"][i] = {}

        # generate lazy zonal statistics, shape metrics for each region within a time slice for all variables
        ar_di[timestamp]["blobs with IVT magnitude"] = regionprops(
            labeled_slice.astype(int).values, intensity_image=ivt_magnitude.data
        )

        ar_di[timestamp]["blobs with IVT poleward"] = regionprops(
            labeled_slice.astype(int).values, intensity_image=ivt_poleward.data
        )

        ar_di[timestamp]["blobs with IVT direction"] = regionprops(
            labeled_slice.astype(int).values, intensity_image=ivt_dir.data
        )
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
        length_width_ratio = blob.axis_major_length / blob.axis_minor_length
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
    # find endpoints of the major axes with respect to the centroid
    # note that we force integers because we ultimately desire array indices
    y0 = int(y0)
    x0 = int(x0)
    x1 = int(x0 - math.sin(orientation) * 0.5 * blob.axis_major_length)
    y1 = int(y0 - math.cos(orientation) * 0.5 * blob.axis_major_length)

    # the ellipse model used to find the orientation and major axis lengths may extend into cartesian space that is beyond
    # the geographic extent of the data, so we will check and correct
    if x1 < 0:
        x1 = 0
    elif x1 >= ds.longitude.shape[0]:
        x1 = ds.longitude.shape[0] - 1
    if y1 < 0:
        y1 = 0
    elif y1 >= ds.latitude.shape[0]:
        y1 = ds.latitude.shape[0] - 1

    # use the array indices to select the latitude and longitude of the centroid and a major axis endpoint
    centroid_lat = ds.latitude[y0].values
    centroid_lon = ds.longitude[x0].values
    geo_centroid = (centroid_lat, centroid_lon)
    major_lat = ds.latitude[y1].values
    major_lon = ds.longitude[x1].values
    geo_major_axis_endpoint = (major_lat, major_lon)

    # total major axis length will be twice the distance between these points
    # haversine distance used because we expect points to be substantially far apart
    half_major_axis_length = haversine(geo_centroid, geo_major_axis_endpoint)
    # km is default unit for haversine function
    major_axis_length_km = round(half_major_axis_length * 2)
    return blob.label, major_axis_length_km


def get_azimuth_of_furthest_points(blob, ds):
    """
    Compute the forward azimuth of the major axis of a labeled region.

    Parameters
    ----------
    blob : skimage.measure._regionprops.RegionProperties
        Region properties object representing a labeled region.
    ds : xarray.Dataset
        must have latitude and longitude coordinates.

    Returns
    -------
    tuple
        (label of the region, forward azimuth of the major axis)

    Notes
    -----
    The function calculates the forward azimuth of the major axis of a labeled region using the region's centroid and orientation properties. It also requires the input dataset ('ds') containing latitude and longitude
    coordinates for spatial reference.

    The major axis is defined by three points: the region's centroid and two endpoints that extend half the length of the major axis from the centroid along the region's orientation. The function calculates the latitude and longitude of the end points using the orientation and axis lengths provided by the 'blob' object.

    The coordinates of the major axis endpoints are then used to compute the geographic forward azimuth of the region.
    """
    y0, x0 = blob.centroid
    orientation = blob.orientation
    # find endpoints of the major axis with respect to the centroid
    # note that we force integers because we ultimately desire array indices
    y0 = int(y0)
    x0 = int(x0)
    x1 = int(x0 - math.sin(orientation) * 0.5 * blob.axis_major_length)
    y1 = int(y0 - math.cos(orientation) * 0.5 * blob.axis_major_length)
    x2 = int(x0 + math.sin(orientation) * 0.5 * blob.axis_major_length)
    y2 = int(y0 + math.cos(orientation) * 0.5 * blob.axis_major_length)

    # the ellipse model used to find the orientation and major axis lengths may extend into cartesian space that is beyond
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

    # use the array indices to select the latitude and longitude of the major axis endpoints
    major_lat1 = ds.latitude[y1].values
    major_lon1 = ds.longitude[x1].values
    major_lat2 = ds.latitude[y2].values
    major_lon2 = ds.longitude[x2].values

    # calculate the forward azimuth using WGS84 geoid and the inv function
    # for the northern hemisphere, the function needs the more southerly coords first
    geodesic = pyproj.Geod(ellps="WGS84")
    fwd_azimuth, _back_azimuth, _distance = geodesic.inv(
        major_lon2, major_lat2, major_lon1, major_lat1
    )

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


def get_total_ivt_strength(blob):
    """
    Computes the total strength of a labeled region as the regional sum of IVT.

    Parameters
    ----------
    blob : skimage.measure._regionprops.RegionProperties
        Region properties object representing a labeled region.

    Returns
    -------
    tuple
        (label of the region, total IVT strength)
    """
    return blob.label, round(blob.image_intensity.sum())


def get_relative_ivt_strength(blob):
    """
    Computes the relative strength of a labeled region as the regional sum of IVT divided by region area.

    Parameters
    ----------
    blob : skimage.measure._regionprops.RegionProperties
        Region properties object representing a labeled region.

    Returns
    -------
    tuple
        (label of the region, relative IVT strength)
    """
    return blob.label, round(blob.image_intensity.sum() / blob.area)


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
            int_label, major_axis_length_km = get_major_axis_haversine_distance(
                blob, ds
            )
            ar_di[k]["ar_targets"][int_label][
                "major axis length (km)"
            ] = major_axis_length_km

    for k in tqdm(
        ar_di, desc="Getting overall orientation (azimuth) for each AR target:"
    ):
        for blob in ar_di[k]["blobs with IVT magnitude"]:
            int_label, max_distance_mean_azimuth = get_azimuth_of_furthest_points(
                blob, ds
            )
            ar_di[k]["ar_targets"][int_label][
                "overall orientation"
            ] = max_distance_mean_azimuth

    for k in tqdm(ar_di, desc="Getting mean poleward strength for each AR target:"):
        for blob in ar_di[k]["blobs with IVT poleward"]:
            int_label, mean_poleward_strength = get_poleward_strength(blob)
            ar_di[k]["ar_targets"][int_label][
                "mean poleward strength"
            ] = mean_poleward_strength

    for k in tqdm(ar_di, desc="Getting directional coherence for each AR target:"):
        for blob in ar_di[k]["blobs with IVT direction"]:
            int_label, pct_coherent, mean_of_ivt_dir = get_directional_coherence(blob)
            ar_di[k]["ar_targets"][int_label]["directional_coherence"] = pct_coherent
            ar_di[k]["ar_targets"][int_label]["mean_of_ivt_dir"] = round(
                mean_of_ivt_dir
            )

    for k in tqdm(ar_di, desc="Getting total IVT strength for each AR target:"):
        for blob in ar_di[k]["blobs with IVT magnitude"]:
            int_label, total_ivt_strength = get_total_ivt_strength(blob)
            ar_di[k]["ar_targets"][int_label]["total ivt strength"] = total_ivt_strength

    for k in tqdm(ar_di, desc="Getting relative IVT strength for each AR target:"):
        for blob in ar_di[k]["blobs with IVT magnitude"]:
            int_label, relative_ivt_strength = get_relative_ivt_strength(blob)
            ar_di[k]["ar_targets"][int_label][
                "relative ivt strength"
            ] = relative_ivt_strength

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
    criteria = [
        "Coherence in IVT Direction",
        "Mean Meridional IVT",
        "Consistency Between Mean IVT Direction and Overall Orientation",
        "Length",
        "Length/Width Ratio",
    ]

    for k in tqdm(ar_di):
        for blob_label in ar_di[k]["ar_targets"]:
            for criterion in criteria:
                ar_di[k]["ar_targets"][blob_label][criterion] = False
            # Length / width ratio criterion
            if ar_di[k]["ar_targets"][blob_label]["length/width ratio"] > 2:
                ar_di[k]["ar_targets"][blob_label]["Length/Width Ratio"] = True
            # Axis length criterion
            if (
                ar_di[k]["ar_targets"][blob_label]["major axis length (km)"]
                > ar_params["min_axis_length"]
            ):
                ar_di[k]["ar_targets"][blob_label]["Length"] = True
            # Directional coherence criterion
            if ar_di[k]["ar_targets"][blob_label]["directional_coherence"] > 50:
                ar_di[k]["ar_targets"][blob_label]["Coherence in IVT Direction"] = True
            # Strong poleward component criterion
            if (
                ar_di[k]["ar_targets"][blob_label]["mean poleward strength"]
                > ar_params["mean_meridional"]
            ):
                ar_di[k]["ar_targets"][blob_label]["Mean Meridional IVT"] = True
            # Consistency Between Mean IVT Direction and Overall Orientation criterion
            if (
                abs(
                    180
                    - (
                        180
                        - ar_di[k]["ar_targets"][blob_label]["mean_of_ivt_dir"]
                        + ar_di[k]["ar_targets"][blob_label]["overall orientation"]
                    )
                    % 360
                )
                < ar_params["orientation_deviation_threshold"]
            ):
                ar_di[k]["ar_targets"][blob_label][
                    "Consistency Between Mean IVT Direction and Overall Orientation"
                ] = True

            # how many criteria were met?
            ar_di[k]["ar_targets"][blob_label]["Criteria Passed"] = list(
                map(ar_di[k]["ar_targets"][blob_label].get, criteria)
            ).count(True)
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
        contains timestamps as keys and region IDs as a list of values for the ARs that meet the specified number of criteria

    Notes
    -----
    The function iterates through each time slice in 'ar_di' and checks each labeled region within that time slice. If a region has met the required number of criteria (determined by 'n_criteria_required'), it is added to the 'filtered_ars' dictionary along with the timestamp and region ID.
    """
    filtered_ars = {}
    for k in tqdm(ar_di):
        passing_blobs_list = []

        for blob_label in ar_di[k]["ar_targets"]:
            if (
                ar_di[k]["ar_targets"][blob_label]["Criteria Passed"]
                >= n_criteria_required
            ):
                passing_blobs_list.append(blob_label)

        if len(passing_blobs_list) > 0:
            filtered_ars[k] = passing_blobs_list

    return filtered_ars


def create_geodataframe_with_all_ars(filtered_ars, ar_di, labeled_blobs, ivt_ds):
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
    ivt_ds : xarray.Dataset
        The IVT dataset (used to provide affine transform, so must contain a CRS).

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing polygons representing all ARs meeting the criteria.

    Notes
    -----
    Attributes of each AR, such as IVT strength, mean IVT, max IVT, min IVT, and time, are added as columns to the GeoDataFrame.
    """

    crs = str(ivt_ds.rio.crs)
    aff = ivt_ds.sel(time=str(ivt_ds.time[0].values)).rio.transform()

    gdfs = []

    for k in tqdm(filtered_ars):
        l = labeled_blobs.sel(time=k)

        r = shapes(l, mask=l.isin(filtered_ars[k]), connectivity=8, transform=aff)
        ar_polys = list(r)

        blob_geom = [shape(i[0]) for i in ar_polys]
        blob_geom = gpd.GeoSeries(blob_geom, crs=crs)

        blob_labels = [i[1] for i in ar_polys]
        blob_labels = pd.Series(blob_labels)

        blob_props = [ar_di[k]["ar_targets"][i] for i in blob_labels]

        result = gpd.GeoDataFrame(
            {
                "time": k,
                "blob_label": blob_labels,
                "blob_props": blob_props,
                "geometry": blob_geom,
            }
        )

        m = result.blob_props.apply(pd.Series)
        new_cols = m.columns.values.tolist()
        result[new_cols] = m
        result.drop("blob_props", axis=1, inplace=True)

        gdfs.append(result)

    all_ars = pd.concat(gdfs)

    all_ars.reset_index(inplace=True, drop=True)

    return all_ars


def create_shapefile(all_ars, shp_fp, csv_fp):
    """
    Save a shapefile to disk from the GeoDataFrame containing all ARs meeting the criteria. Column names are abbreviated to 10 characters or less, and a .csv file is output with full column name descriptions.

    Parameters
    ----------
    all_ars : geopandas.GeoDataFrame
        GeoDataFrame containing polygons representing all ARs meeting the criteria.
    shp_fp : Posix path
        File path of output shapefile
    csv_fp : Posix path
        File path of output csv

    Returns
    -------
    None
    """
    old_cols = all_ars.columns.to_list()
    new_cols = [
        "time",
        "label",
        "geometry",
        "ratio",
        "length",
        "orient",
        "poleward",
        "dir_coher",
        "mean_dir",
        "tot_str",
        "rel_str",
        "crit1",
        "crit2",
        "crit3",
        "crit4",
        "crit5",
        "crit_cnt",
    ]
    col_dict = dict(zip(old_cols, new_cols))
    all_ars.rename(columns=col_dict, inplace=True)
    #export shp
    all_ars.to_file(shp_fp)

    #set up col descriptions for csv export
    desc = [
        'timestep of AR',
        'original candidate region label of timestep AR',
        'geometry string for AR polygon',
        'length to width ratio of timestep AR',
        'length (km) of timestep AR',
        'orientation of timestep AR',
        'poleward strength of timestep AR',
        'directional coherence (%) of timestep AR',
        'mean IVT direction of timestep AR',
        'sum of IVT within timestep AR',
        'sum of relative IVT (sum IVT/area) within timestep AR',
        'Coherence in IVT direction (1 = True / 0 = False)',
        'Mean Meridional IVT (1 = True / 0 = False)',
        'Consistency Between Mean IVT Direction and Overall Orientation (1 = True / 0 = False)',
        'Length (1 = True / 0 = False)',
        'Length/Width Ratio (1 = True / 0 = False)',
        'Number of criteria passed'
    ]
    csv_dict = dict(zip(new_cols, desc))
    pd.DataFrame.from_dict(data=csv_dict, orient="index").reset_index().to_csv(csv_fp, header=['shp_col', 'desc'], index=False)

def landfall_ars_export(shp_fp, csv_fp, ak_shp, landfall_shp, landfall_csv, landfall_events_shp, landfall_events_csv):
    """Filter the raw AR detection shapefile output to include only ARs making landfall in Alaska, and export the result to a new shapefile. Process the landfall ARs to condense adjacent dates into event multipolygons, and export the result to a second new shapefile. For both outputs, include a CSV with column names and descriptions.

    Parameters
    ----------
    shp_fp : PosixPath
        File path to the raw AR detection shapefile input.
    csv_fp : PosixPath
        File path to the raw AR detection csv input.   
    ak_shp : PosixPath
        File path to the Alaska coastline shapefile input.
    landfall_shp : PosixPath
        File path for the landfalling AR shapefile output.
    landfall_csv : PosixPath
        File path for the landfalling AR csv output.
    landfall_events_shp : PosixPath
        File path for the condensed landfall AR events shapefile output.
    landfall_events_csv : PosixPath
        File path for the landfalling AR events csv output.
        

    Returns
    -------
    None
    """

    # import AK coastline and AR shps to gdfs
    ars = gpd.read_file(shp_fp)
    ak = gpd.read_file(ak_shp)

    # reproject AK coastline to match AR CRS, and dissolve into one multipolygon
    ak_ = ak.to_crs(ars.crs)
    ak_d = ak_.dissolve()

    # add new datetime column to ars gdf by parsing ISO timestamp
    # reformat time column string for output (datetime fields not supported in ESRI shp files)
    ars["dt"] = [
        datetime.fromisoformat(ars.time.iloc[[d]].values[0][:-16])
        for d in range(len(ars))
    ]
    ars["time"] = ars["dt"].astype(str)

    # perform spatial join, keeping only AR polygons that intersect with the AK polygon
    ak_ars = ars.sjoin(ak_d, how="inner", predicate="intersects")

    # export landfall geodataframe to shp
    ak_ars.drop(columns=["dt", "index_right", "FEATURE"]).to_file(landfall_shp, index=True)
    # copy original AR detection csv to the landfall fp (these two outputs have the same exact fields)
    t = pd.read_csv(csv_fp)
    newrow = ["index", "table index value in raw AR detection shapefile"]
    t.loc[len(t)] = newrow
    t.to_csv(landfall_csv, index=False)

    # wrap the circular mean function using max 360 arg
    def circ_mean(x):
        return circmean(x, high=360)

    # label any ARs occuring on adjacent dates with a unique "diff" ID and inspect as a subset dfs
    # use connected components of  a matrix to label the spatially overlapping groups of polygons in each subset df
    # (this allows for separation of multiple non-overlapping AR events occuring in the same time period)
    # dissolve geometry by group and aggregate values into new columns; calculate new properties before concat and export

    ak_ars["diff"] = ak_ars["dt"].diff().dt.days.gt(1).cumsum()

    dfs = []

    for d in ak_ars["diff"].unique():
        sub = ak_ars[ak_ars["diff"] == d].copy()

        # spatial overlap analysis of adjacent date subset
        overlap_matrix = sub.geometry.apply(
            lambda x: sub.geometry.intersects(x)
        ).values.astype(int)
        n, labels = connected_components(overlap_matrix)
        sub["group"] = labels
        sub["start"] = sub["dt"]
        sub["end"] = sub["dt"]
        sub["sumtot_str"] = sub["tot_str"]
        sub["sumrel_str"] = sub["rel_str"]
        sub["ratio_m"] = sub["ratio"]
        sub["len_km_m"] = sub["length"]
        sub["orient_m"] = sub["orient"]
        sub["poleward_m"] = sub["poleward"]
        sub["dircoher_m"] = sub["dir_coher"]
        sub["mean_dir_m"] = sub["mean_dir"]

        # dissolve geometry and aggregate
        res = sub.dissolve(
            by="group",
            aggfunc={
                "start": "min",
                "end": "max",
                "sumtot_str": "sum",
                "sumrel_str": "sum",
                "ratio_m": "mean",
                "len_km_m": "mean",
                "orient_m": circ_mean,
                "poleward_m": "mean",
                "dircoher_m": "mean",
                "mean_dir_m": circ_mean,
            },
        )

        # calculate duration from datetime columns
        for i in res.index:
            # after subtracting, add 6hrs as minimum event length.... this assures a single timestep event is not zero duration!
            res.loc[i, "dur_hrs"] = (
                (res["end"][i] - res["start"][i]).total_seconds() / 3600
            ) + 6

        # calculate total and relative intensity
        res["tintensity"] = res["sumtot_str"] / res["dur_hrs"]
        res["rintensity"] = res["sumrel_str"] / res["dur_hrs"]

        # round results
        res = res.round(
            {
                "ratio_m": 0,
                "len_km_m": 0,
                "orient_m": 0,
                "poleward_m": 0,
                "dircoher_m": 0,
                "mean_dir_m": 0,
                "dur_hrs": 0,
                "tintensity": 0,
                "rintensity": 0,
            }
        )

        dfs.append(res)

    events = pd.concat(dfs)
    events = events.reset_index(drop=True).reset_index(names='event_id')
    events.crs = ars.crs

    # reset datetime columns as strings for output (datetime fields not supported in ESRI shp files)
    events["start"] = events["start"].astype(str)
    events["end"] = events["end"].astype(str)

    # export condensed event AR geodataframe to shp
    events.to_file(landfall_events_shp, index=False)

    # set up AR events columns decription table
    cols = events.columns.to_list()
    desc = [
        'unique AR event ID',
        'geometry string for AR event polygons',
        'first timestep of AR event',
        'last timestep of AR event',
        'sum of IVT across all timestep ARs in event',
        'sum of relative IVT (sum IVT/area) across all timestep ARs in event',
        'mean length to width ratio across all timestep ARs in event',
        'mean length (km) across all timestep ARs in event',
        'mean orientation across all timestep ARs in event',
        'mean poleward strength across all timestep ARs in event',
        'mean directional coherence (%) across all timestep ARs in event',
        'mean IVT direction across all timestep ARs in event',
        'duration of AR event',
        'sum of AR event total intensity divided by AR event duration',
        'sum of AR event relative intensity divided by AR event duration'
        ]

    csv_dict = dict(zip(cols, desc))
    # export event AR column description table to csv
    pd.DataFrame.from_dict(data=csv_dict, orient="index").reset_index().to_csv(landfall_events_csv, header=['shp_col', 'desc'], index=False)

def detect_all_ars(fp, n_criteria, out_shp, out_csv, ak_shp, landfall_shp, landfall_csv, landfall_events_shp, landfall_events_csv):
    """Run the entire AR detection pipeline and generate shapefile output.

    Parameters
    ----------
    fp : Posix path
        File path to the IVT dataset in NetCDF format.
    n_criteria : int
        The number of criteria required to consider a region an AR.
    out_shp : Posix path
        File path to save the shapefile output.
    out_csv : Posix Path
        File path to save the csv output.
    ak_shp : Posix Path
        File path of Alaska coastline shapefile input.
    landfall_shp : PosixPath
        File path for the landfalling AR shapefile output.
    landfall_csv : PosixPath
        File path for the landfalling AR csv output.
    landfall_events_shp : PosixPath
        File path for the condensed landfall AR events shapefile output.
    landfall_events_csv : PosixPath
        File path for the landfalling AR events csv output.

    Returns
    -------
    None
    """
    with xr.open_dataset(fp) as ivt_ds:
        ivt_ds.rio.write_crs("epsg:4326", inplace=True)
        ivt_ds["thresholded"] = compute_intensity_mask(
            ivt_ds["ivt_mag"], ivt_ds["ivt_quantile"], ar_params["ivt_floor"]
        )
        labeled_regions = label_contiguous_mask_regions(ivt_ds["thresholded"])
        ar_di = generate_region_properties(labeled_regions, ivt_ds)
        ar_di = get_data_for_ar_criteria(ar_di, ivt_ds)
        ar_di = apply_criteria(ar_di)
        output_ars = filter_ars(ar_di, n_criteria_required=n_criteria)
        output_ar_gdf = create_geodataframe_with_all_ars(
            output_ars, ar_di, labeled_regions, ivt_ds
        )
        create_shapefile(output_ar_gdf, out_shp, out_csv)
        landfall_ars_export(out_shp, out_csv, ak_shp, landfall_shp, landfall_csv, landfall_events_shp, landfall_events_csv)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect atmospheric rivers and generate shapefile output."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=ard_fp,
        help="File path to the IVT dataset in NetCDF format.",
    )
    parser.add_argument(
        "--n_criteria",
        type=int,
        default=5,
        help="Number criteria required to consider a region an AR.",
    )
    parser.add_argument(
        "--output_ar_shapefile",
        type=str,
        default=shp_fp,
        help="File path to save the full AR shapefile output.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=csv_fp,
        help="File path to save the CSV output.",
    )
    parser.add_argument(
        "--ak_shapefile",
        type=str,
        default=ak_shp,
        help="File path of AK shapefile input.",
    )
    parser.add_argument(
        "--output_landfall_shapefile",
        type=str,
        default=landfall_shp,
        help="File path to save the landfall AR shapefile output.",
    )
    parser.add_argument(
        "--output_landfall_csv",
        type=str,
        default=landfall_csv,
        help="File path to save the landfall AR csv output.",
    )
    parser.add_argument(
        "--output_events_shapefile",
        type=str,
        default=landfall_events_shp,
        help="File path to save landfall AR events shapefile output.",
    )
    parser.add_argument(
        "--output_events_csv",
        type=str,
        default=landfall_events_csv,
        help="File path to save landfall AR events csv output.",
    )

    args = parser.parse_args()

    print(
        f"Using the IVT input {args.input_file} to filter candidate ARs based on {args.n_criteria} criteria."
    )

    detect_all_ars(
        fp=args.input_file,
        n_criteria=args.n_criteria,
        out_shp=args.output_ar_shapefile,
        out_csv=args.output_csv,
        ak_shp=args.ak_shapefile,
        landfall_shp=args.output_landfall_shapefile,
        landfall_csv=args.output_landfall_csv,
        landfall_events_shp=args.output_events_shapefile,
        landfall_events_csv=args.output_events_csv
    )

    print(
        f"Processing complete! Full shapefile output written to {args.output_ar_shapefile}, with companion CSV file written to {args.output_csv}. All Alaska landfall ARs shapefile output written to {args.output_landfall_shapefile}, with companion CSV file written to {args.output_landfall_csv}. Condensed Alaska landfall ARs shapefile output written to {args.output_events_shapefile}, with companion CSV file written to {args.output_events_csv}."
    )
