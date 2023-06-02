# this AR detection script will store a set of functions designed to be run from a notebook.
import xarray as xr
import numpy as np
import rioxarray
import rasterio
from scipy.ndimage import label, generate_binary_structure, labeled_comprehension
from skimage.measure import regionprops



# IVT intensity masking
# use to identify where IVT magnitude > IVT percentile and IVT magnitude > IVT floor
# values meeting criteria will keep their IVT magnitude values
# values not meeting criteria will be assigned NA
def intensity_mask(ivt_mag, ivt_pctile, ivt_floor):
    func = lambda x, y, z: xr.where((x>y) & (x>z), x, 0)
    return xr.apply_ufunc(func, ivt_mag, ivt_pctile, ivt_floor, dask='parallelized')

# Labeling and cataloging continuous geometric regions
# regions will be labeled with a unique numeric identifier at each timestep
# output will be a data cube with the same dimensions as the intensity mask input
def label_regions(mask):     
    # set search structure to find contiguous regions...this structure will include diagonal neighbors
    s = generate_binary_structure(2,2)
    
    # copy mask data array to new da variable, and also:
    # rename the variable
    # assign all values as NA
    # this is now a template data array with same size/dimensions as the input to store region labeling results
    
    da = mask.copy(deep=False).rename('regions').where(mask.values == -9999)
    
    # loop thru input mask data array over time dimension
    # label continuous regions using search structure for each timestep
    # assign the new region array values to the proper timestep in the empty template array

    for a in range(len(mask.time)):
        labeled_array, num_features = label(mask[a].values, structure=s)
        da[a] = labeled_array
 
    return da




# filtering by geometry
# this function will alter the labeled regions data array in place, removing labels that dont meet shape criteria
# if they have labeled regions, reproject individual 2D region arrays to a 1km grid;
# (image measurements are given in "pixel" units, so with a 1km grid we can directly compare shape criteria in km)
# apply regionprops function to array at each timestep;
# identify any region label where properties do not meet provided minimum length criteria and minimum 2:1 length/width ratio criteria
# create drop dictionary with timestep as key and labels as values
# use the drop dictionary to do another loop thru the original dataset and reassign dropped labels to 0
# ***** Note that since this function uses the entire spatial domain for geometric shape detection, we should not use dask chunking along x/y dimensions!

def filter_regions_by_geometry(regions, min_axis_length):
    # establish CRS as WGS84
    regions.rio.write_crs("epsg:4326", inplace=True)
    
    drop_dict = {}
    
    for a in regions:
        
        # check if there are actually any labeled regions, to avoid reprojections on empty arrays
        # arrays with no regions should only have one unique value (0)
        if len(np.unique(a.values)) > 1:
        
            #reproject to epsg:3338 with 1000m (1km) grid cells
            # no data is 0 since this is a categorical raster where all non-zero values are region labels
            a_3338 = a.rio.reproject('epsg:3338', resolution=1000, nodata=0, resampling=rasterio.enums.Resampling.nearest)

            props = regionprops(a_3338.astype(int).values)

            drop_list = []

            for p in props:
                #check axis length
                if p.major_axis_length < min_axis_length:
                    drop_list.append(p.label)
                #check length to width ratio 2:1 or greater    
                #elif (p.major_axis_length / p.minor_axis_length) < 2:
                    #drop_list.append(p.label)
            
            if len(drop_list) > 0:
                drop_dict[a_3338.time.values] = drop_list
           
    for d in drop_dict: 
        regions.loc[dict(time=d)]  = xr.where(regions.sel(time=d).isin(drop_dict[d]), 0, regions.sel(time=d))
         
    return drop_dict






# filtering by AR direction  --- **** INCOMPLETE!!! ****
# this function will alter the labeled regions data array in place, removing labels that dont meet shape criteria

# there are 3 components to this function, following the rules of G&W 2015 (https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2015JD024257)

# 1. "Coherence in IVT Direction. If more than half of the grid cells have IVT deviating more than 45° from the object's mean IVT, the object is discarded."
# STEPS: 
#  - list the ivt_dir array values from labeled comprehension
#  - count values that differ more than 45deg from mean ivt direction just computed
#  - if >50% values meet that criteria, return a simple "drop" or "keep" string
# NOTE: this labeled comprehension returns an array with one element for each label: you cannot return lists of variable length...
# that always results in a "ValueError: setting an array element with a sequence" failure
# this is why the function currently prints a list, but cannot return that list...
# instead we see array of [nan nan] below the printed list output)
#  - use the timesteps, labels, and drop/keep outputs to populate the drop_dict

# 2. "Object Mean Meridional IVT. ... an object is discarded if the mean IVT does not have an appreciable poleward component (>50 kg m−1 s−1)."
# this poleward component is the p72.162 variable in the ERA5 dataset, equivalent to northward "v" component of IVT
# using labeled comprehension, the code would look something like this:
# for r, p in zip(regions, ivt_poleward):
#     lbls = list(np.unique(r.values))
#     mean_poleward = labeled_comprehension(p, r, lbls, np.mean, float, 0)

# 3. "Consistency Between Object Mean IVT Direction and Overall Orientation...An object is discarded if the direction of mean IVT deviates from the overall orientation by more than 45°." G&W use the azimuth of the line drawn at maximum great circle distance of the labeled region. I think we would find the two cells within the labeled region that are at maximum distance from each other, find the line between them, and get its azimuth to compare to the mean direction calculated in step 1. NOTE that the degree directions used in step 1 represent direction the AR is coming FROM....so the azimuth here would also need to point in the FROM direction, not the TO direction....


#  LAST STEP:
# - use drop_dict similar to filter_regions_by_geometry() function to remove region labels in place



    
def filter_regions_by_direction(regions, ivt_dir):
    
    drop_dict={}
    
    
    # 1. Coherence in IVT Direction
    
    # custom function, needs work! currently just prints a list
    def count_wind_dir(vals):
        l=[]
        for v in vals:
            l.append(v)
        print(l)
    
    
    for r, d in zip(regions, ivt_dir):
        
        # check if there are actually any labeled regions, to avoid math on empty arrays
        # arrays with no regions should only have one unique value (0)
        if len(np.unique(r.values)) > 1:
            
            # get unique labels
            lbls = list(np.unique(r.values))
            print("Labeled regions at " + str(r.time.values) + ": " + str(lbls))
            
            # use label comprehension to get mean ivt_dir for each labeled region
            mean_dirs = labeled_comprehension(d, r, lbls, np.mean, float, 0)
            print("Mean IVT directions at " + str(r.time.values) + ": " + str(mean_dirs))
            
            # apply custom function to compare ivt_dir values in labeled areas to means
            drop_keep = labeled_comprehension(d, r, lbls, count_wind_dir, float, 0)
            print(drop_keep)
            
            # populate the drop_dict
            #for d in drop_dict: 
                #regions.loc[dict(time=d)] = xr.where(regions.sel(time=d).isin(drop_dict[d]), 0, regions.sel(time=d))
    
    # 2. Object Mean Meridional IVT
    # 3. Consistency Between Object Mean IVT Direction and Overall Orientation
    
    
    
    #remove regions in place if they are in the drop dict
    #for d in drop_dict: 
        #regions.loc[dict(time=d)]  = xr.where(regions.sel(time=d).isin(drop_dict[d]), 0, regions.sel(time=d))
         
    
    
    return drop_dict




#compute_ar_stats("AR", IVT_data_cube, "direction", "shape")
#compute general stats for all cells within final AR objects (eg min/mean/max IVT, direction, etc)
#compute basic non-stat values for final AR objects (eg length, width, area, date/time, unique AR id?)
#return long format "stats_df" dataframe with columns (AR / stat / value)


#ar_to_gdf("AR", "stats_df")
#convert all ARs to polygon geodataframe
#join AR stats as attributes
#(polygons will be overlapping in the case of long duration AR events)
#return geodataframe
