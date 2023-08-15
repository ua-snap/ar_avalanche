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
from haversine import haversine
from shapely.geometry import Polygon, shape
from rasterio.features import shapes
from datetime import datetime


from config import ar_params, ard_fp, shp_fp, csv_fp, ak_shp, landfall_6hr_fp, landfall_events_fp


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

    upper_lat = ds.latitude[minrow].values
    left_lon = ds.longitude[mincol].values
    lower_lat = ds.latitude[maxrow].values
    right_lon = ds.longitude[maxcol].values
    
    # JP: the lats below were reversed, which caused a backwards bbox and the forward azimuth result to be reversed...keeping original below for reference
    # lower_lat = ds.latitude[minrow].values
    # left_lon = ds.longitude[mincol].values
    # upper_lat = ds.latitude[maxrow].values
    # right_lon = ds.longitude[maxcol].values
    
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
            # Length / width ratio criterion
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
            ar_di[k]["ar_targets"][blob_label]["Criteria Passed"] = list(map(ar_di[k]["ar_targets"][blob_label].get, criteria)).count(True)
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
            if ar_di[k]["ar_targets"][blob_label]["Criteria Passed"] >= n_criteria_required:
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
    Attributes of each AR, such as mean IVT, max IVT, min IVT, and time, are added as columns to the GeoDataFrame.
    """

    crs = str(ivt_ds.rio.crs)
    aff = ivt_ds.sel(time=str(ivt_ds.time[0].values)).rio.transform()

    gdfs = []

    for k in tqdm(filtered_ars):

        l = labeled_blobs.sel(time=k)
                
        r = shapes(l, mask = l.isin(filtered_ars[k]), connectivity=8, transform=aff)
        ar_polys = list(r)

        blob_geom = [shape(i[0]) for i in ar_polys]
        blob_geom = gpd.GeoSeries(blob_geom, crs=crs)

        blob_labels = [i[1] for i in ar_polys]
        blob_labels = pd.Series(blob_labels)

        blob_props = [ar_di[k]['ar_targets'][i] for i in blob_labels]     

        result = gpd.GeoDataFrame({'time': k, 'blob_label': blob_labels, 'blob_props': blob_props, 'geometry': blob_geom})
        
        m = result.blob_props.apply(pd.Series)
        new_cols = m.columns.values.tolist()
        result[new_cols] = m
        result.drop('blob_props', axis=1, inplace=True)

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
    new_cols = ['time', 'label', 'geometry', 'ratio', 'length', 'orient', 'poleward', 'dir_coher', 'mean_dir', 'crit1', 'crit2', 'crit3', 'crit4', 'crit5', 'crit_cnt']
    col_dict = dict(zip(old_cols,new_cols))
    all_ars.rename(columns=col_dict, inplace=True)

    all_ars.to_file(shp_fp)

    pd.DataFrame.from_dict(data=col_dict, orient='index').to_csv(csv_fp, header=False)


def landfall_ars_export(shp_fp, ak_shp, fp_6hr, fp_events):
    """Filter the raw AR detection shapefile output to include only ARs making landfall in Alaska, and export the result to a new shapefile. Process the landfall ARs to condense adjacent dates into multipolygons, and export the result to a second new shapefile.

    Parameters
    ----------
    shp_fp : PosixPath
        File path to the raw AR detection shapefile input.
    ak_shp : PosixPath
        File path to the Alaska coastline shapefile input.
    fp_6hr : PosixPath
        File path for the raw 6hr interval landfall AR shapefile output.
    fp_events : PosixPath
        File path for the condensed landfall AR events shapefile output.

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
    ars['dt'] = [datetime.fromisoformat(ars.time.iloc[[d]].values[0][:-16]) for d in range(len(ars))]
    ars['time'] = ars['dt'].astype(str)

    # perform spatial join, keeping only AR polygons that intersect with the AK polygon
    ak_ars = ars.sjoin(ak_d, how='inner', predicate='intersects')

    # export raw 6hr geodataframe to shp
    ak_ars.drop(columns=['dt', 'index_right', 'FEATURE']).to_file(fp_6hr, index=True)

    # label any ARs occuring on adjacent dates with a unique "diff" ID
    # for each ID, turn first and last timestep into gdf attributes and combine their geometry into one multipolygon feature

    ak_ars['diff'] = ak_ars['dt'].diff().dt.days.gt(1).cumsum()

    dfs = []

    for d in ak_ars['diff'].unique():
        sub = ak_ars[ak_ars['diff'] == d]
        geo = shapely.multipolygons([sub.geometry])
        df = gpd.GeoDataFrame({
            'event_id':d,
            'start':sub['dt'].min(),
            'end':sub['dt'].max(),
            'geometry':geo
            })
        dfs.append(df)

    events = pd.concat(dfs)
    events.set_index('event_id', inplace=True)
    events.crs = ars.crs

    # reset datetime columns as strings for output (datetime fields not supported in ESRI shp files)
    events['start'] = events['start'].astype(str)
    events['end'] = events['end'].astype(str)

    # export condensed event AR geodataframe to shp
    events.to_file(fp_events, index=True)


def detect_all_ars(fp, n_criteria, out_shp, out_csv, ak_shp, fp_6hr, fp_events):
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
        output_ar_gdf = create_geodataframe_with_all_ars(output_ars, ar_di, labeled_regions, ivt_ds)
        create_shapefile(output_ar_gdf, out_shp, out_csv)
        landfall_ars_export(out_shp, ak_shp, fp_6hr, fp_events)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect atmospheric rivers and generate shapefile output.")
    parser.add_argument("--input_file", type=str, default=ard_fp, help="File path to the IVT dataset in NetCDF format.")
    parser.add_argument("--n_criteria", type=int, default=5, help="Number criteria required to consider a region an AR.")
    parser.add_argument("--output_ar_shapefile", type=str, default=shp_fp, help="File path to save the full AR shapefile output.")
    parser.add_argument("--output_csv", type=str, default=csv_fp, help="File path to save the CSV output.")
    parser.add_argument("--ak_shapefile", type=str, default=ak_shp, help="File path of AK shapefile input.")
    parser.add_argument("--output_6hr_shapefile", type=str, default=landfall_6hr_fp, help="File path to save the 6hr landfall AR shapefile output.")
    parser.add_argument("--output_events_shapefile", type=str, default=landfall_events_fp, help="File path to save landfall AR events shapefile output.")


    args = parser.parse_args()

    print(f"Using the IVT input {args.input_file} to filter candidate ARs based on {args.n_criteria} criteria.")
    
    detect_all_ars(fp=args.input_file, n_criteria=args.n_criteria, out_shp=args.output_ar_shapefile, out_csv=args.output_csv, ak_shp=args.ak_shapefile, fp_6hr=args.output_6hr_shapefile, fp_events=args.output_events_shapefile)
    
    print(f"Processing complete! Full shapefile output written to {args.output_ar_shapefile}, with companion CSV file written to {args.output_csv}. All Alaska landfall ARs shapefile output written to {args.output_6hr_shapefile}. Condensed Alaska landfall ARs shapefile output written to {args.output_events_shapefile}.")
    
