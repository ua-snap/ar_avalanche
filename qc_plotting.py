"""This module provides helper functions to QC and plot atmospheric river detection outputs."""

import geopandas as gpd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from shapely.geometry import Polygon 
import math
import numpy as np

from config import start_year, end_year, bbox


def attr_check(shp, tbl):
    """Compare attributes of output shapefiles and output CSV tables. Prints a message indicating whether they are identical.

    Parameters
    ----------
    shp : shapefile path
        AR detection shapefile output
    tbl : csv path
        AR detection csv output

    Returns
    -------
    None
    """
    s = gpd.read_file(shp)
    t = pd.read_csv(tbl)

    pd.set_option('display.max_colwidth', None)
    display(t)
    pd.set_option('display.max_colwidth', 150)

    missing = []
    for a in s.columns.values.tolist():
        if a not in t.shp_col.values.tolist():
            missing.append(a)
    
    if len(missing) == 0:
        print("QC passed! Columns in shapefile and table are identical.")
    else:
        print("QC failed! There are some missing columns: ")
        print(missing)


def count_ars(raw, landfall, events, points):
    """Count ARs in each AR detection output. Print a message describing the spatiotemporal extent and counts.

    Parameters
    ----------
    raw : Posix path
        file path to raw AR detection shapefile output (all ARs)
    landfall : Posix path
        filepath to landfalling AR detection shapefile output (only landfalling ARs)
    events : Posix path
        filepath to aggregated landfalling AR detection shapefile output (only landfalling AR events)
    points : shapefile path
        filepath to landfalling AR coastal impact points shapefile output
    
    Returns
    -------
    None
    """
    r = gpd.read_file(raw)
    l = gpd.read_file(landfall)
    e = gpd.read_file(events)
    p = gpd.read_file(points)

    print("AR detection was performed between " + str(start_year) + " and " + str(end_year) + ", in an area from latitude " + str(bbox[0]) + " to " + str(bbox[2]) + " and longitude " + str(bbox[1]) + " to " + str(bbox[3]) + ".")
    print("\n")
    print("There were " + str(len(r)) + " individual timestep ARs detected across the entire spatial and temporal domain.")
    print("Of these, " + str(len(l)) + " of the detected ARs intersect the Alaska polygon boundary.")
    print("Of these, " + str(len(e)) + " possible AR events were aggregated from landfalling ARs, using a combination of adjacent timesteps and overlapping geometry to define an event.")
    print(str(len(p)) + " AR events intersected the coastline along their primary axis of travel.")
    print(str(len(e) - len(p)) + " AR events made landfall without their primary axis of travel did not intersect the coastline.")
    print("\n")
    print("On average, we detected " + str(len(p)/30) + " yearly AR events with a coastal impact points.")


def create_hexgrid(ak, gdf, side_length):
    """Create a hexagon grid to aggregate AR events points in space. Adapted from https://pygis.io/docs/e_summarize_vector.html.

    Parameters
    ----------
    ak : geodataframe
        geodataframe of Alaska boundary; must have a projected CRS
    gdf : geodataframe
        geodataframe of events layer; must have a projected CRS
    side length : integer
        length of a hexagon side, in meters

    Returns
    -------
    Hexgrid geodataframe with unique integer grid IDs for spatial joining operations.
    """

    # get extents
    min_x1, min_y1, max_x1, max_y1 = gdf.total_bounds
    min_x2, min_y2, max_x2, max_y2 = ak.total_bounds
    # find extent of both layers combined
    min_x = min(min_x1, min_x2)
    min_y = min(min_y1, min_y2)
    max_x = max(max_x1, max_x2)
    max_y = max(max_y1, max_y2)

    # create empty list to hold individual cells that will make up the grid
    cells_list = []
    # Set horizontal displacement that will define column positions with specified side length (based on normal hexagon)
    x_step = 1.5 * side_length
    # Set vertical displacement that will define row positions with specified side length (based on normal hexagon)
    # This is the distance between the centers of two hexagons stacked on top of each other (vertically)
    y_step = math.sqrt(3) * side_length
    # Get apothem (distance between center and midpoint of a side, based on normal hexagon)
    apothem = (math.sqrt(3) * side_length / 2)
    # Set column number
    column_number = 0

    # Create and iterate through list of x values that will define column positions with vertical displacement
    for x in np.arange(min_x, max_x + x_step, x_step):

        # Create and iterate through list of y values that will define column positions with horizontal displacement
        for y in np.arange(min_y, max_y + y_step, y_step):
            # Create hexagon with specified side length
            hexagon = [[x + math.cos(math.radians(angle)) * side_length, y + math.sin(math.radians(angle)) * side_length] for angle in range(0, 360, 60)]
            # Append hexagon to list
            cells_list.append(Polygon(hexagon))

        # Check if column number is even
        if column_number % 2 == 0:
            # If even, expand minimum and maximum y values by apothem value to vertically displace next row
            # Expand values so as to not miss any features near the feature extent
            min_y -= apothem
            max_y += apothem

        # Else, odd
        else:
            # Revert minimum and maximum y values back to original
            min_y += apothem
            max_y -= apothem

        # Increase column number by 1
        column_number += 1

    # Create grid from list of cells
    grid = gpd.GeoDataFrame(cells_list, columns = ['geometry'], crs = gdf.crs)
    # Create a column that assigns each grid a number
    grid["Grid_ID"] = np.arange(len(grid))

    # Return grid
    return grid


def agg_by_hexgrid_cells_and_year_month(hexgrid, hexid, gdf, prop, agg, allow_na):
    """Aggregate AR event properties by hexgrid cells.

    Parameters
    ----------
    hexgrid : geodataframe
        geodataframe of hexagon cells with unique grid IDs
    hexid : string
        name of field in hexgrid geodataframe containing unique grid IDs
    gdf : geodataframe
        AR event geodataframe with point geometry (eg, centroids of polygons or coastal intersection points)
    prop : string
        name of field in AR event geodataframe holding property to aggregate within hexgrid cells (eg, 'duration')
    agg : string representing aggregation function
        name of function used to aggregate properties (eg, 'count', 'mean', 'max', etc.)
    allow_na : boolean
        if True, allow NA values in the outputs; if False, fill NA values with zeroes

    Returns
    -------
    Tuple of:
    a) Hexgrid geodataframe with aggregated values for each unique grid cell.
    b) Year/Month pivot table with aggregated values for each unique year/month combo.
    c) aggregated property (to be passed to heatmap plotting function)
    """
    #spatial joins to hexgrid (ar centroid counts)
    hexjoin = gpd.sjoin(gdf, hexgrid, how='inner', predicate='intersects')
    #groupby and apply function
    hexjoin_ = hexjoin.groupby(hexid).agg({prop:agg})
    # merge back to grid and optionally fill 0s
    hexgrid_ = hexgrid.merge(hexjoin_, on = hexid, how = "left")
    if allow_na == False:
        hexgrid_[prop] = hexgrid_[prop].fillna(0)
    else:
        pass
    # convert result to integer (ignoring NA values)
    hexgrid_[prop] = hexgrid_[prop][~hexgrid_[prop].isna()].astype(int)

    #create year/month pivot table using event start datetimes
    gdf["Year"] = gdf.start.apply(lambda x: x.year)
    gdf["Month"] = gdf.start.apply(lambda x: x.month)
    if allow_na == False:
        pt = gdf.pivot_table(index="Month",columns="Year",values=prop, aggfunc=agg).fillna(0)
    else:
        pt = gdf.pivot_table(index="Month",columns="Year",values=prop, aggfunc=agg)

    return hexgrid_, pt, prop


def plot_3panel_heatmaps(pt, ak, hexdf_full, hexdf_coast, prop, cmap, title, cbar_labels, dev, phenom, phenom_string):
    """Plots a 3-panel figure of heatmaps using aggregated hexgrid and pivot table outputs. Includes a year/month heatmap, polygon centroid heatmap, and coastal intersection heatmap. Optionally, include atmospheric phenomenon data.

    Parameters
    ----------
    pt : dataframe
        Year/Month pivot table with aggregated values for each unique year/month combo.
    ak : geodataframe
        Alaska polygon geodataframe, should match CRS of hexgrid geodataframe inputs.
    hexdf_full : geodataframe
        Hexgrid geodataframe with aggregated values for each unique grid cell; based on polygon centroid geometry.
    hexdf_coast : geodataframe
        Hexgrid geodataframe with aggregated values for each unique grid cell; based on coastal impact point geometry.
    prop : string
        name of AR event property aggregated by hexgrid.
    cmap : string
        name of matplotlib colormap used in the heatmap (see: https://matplotlib.org/stable/gallery/color/colormap_reference.html).
    title : string
        title for figure.
    cbar_labels : list
        list of string labels for the main colorbar eg ['low', 'high'].
    dev : boolean
        if True, this is a deviation map that should have a diverging colormap.
    phenom : string or array
        'None' if there is no data to include, or an array of same shape as pivot table (eg, 12 months x 30 years) used to annotate heatmap cells with atmospheric phenomenon data (eg, ENSO, PDO).

    Returns
    -------
    A 3 panel heatmap figure.
    """
    #figure setup
    fig = plt.figure(figsize=(10,9))
    grid = plt.GridSpec(2, 2, wspace=0.05, hspace=-0.35, height_ratios=[1, 2.5], width_ratios=[1,1.57]) 
    heatmap_ax = fig.add_subplot(grid[0, 0:])
    full_ax = fig.add_subplot(grid[1, 0])
    coast_ax = fig.add_subplot(grid[1, 1])

    #colorbar ranges
    pt_vmax = pt.values.max()
    pt_vmin = 0

    #option to set up diverging colorbars using property values...center just above 0, to allow min values of 0
    if dev==True:
        fulldivnorm=mpl_colors.TwoSlopeNorm(vmin=hexdf_full[prop].min(), vcenter=0.01, vmax=hexdf_full[prop].max())
        coastdivnorm=mpl_colors.TwoSlopeNorm(vmin=hexdf_coast[prop].min(), vcenter=0.01, vmax=hexdf_coast[prop].max())

    full_vmax = hexdf_full[prop].max()
    coast_vmax = hexdf_coast[prop].max()

    #font size master reference
    f = 8

    #transparency master reference
    a = 0.8

    #plotting
    if type(phenom) != str:
        sns.heatmap(pt, annot=phenom, annot_kws={'fontsize':f-2}, fmt='', ax=heatmap_ax, cmap=cmap, cbar=False, vmax=pt_vmax, vmin=pt_vmin, alpha=a)
    else: 
        sns.heatmap(pt, fmt='', ax=heatmap_ax, cmap=cmap, cbar=False, vmax=pt_vmax, vmin=pt_vmin, alpha=a)

    ak.geometry.boundary.plot(ax=full_ax, color='black', zorder=1)
    if dev==True:
        hexdf_full.plot(ax=full_ax, column=prop, cmap=cmap, norm=fulldivnorm, legend=False, edgecolor='lightgray', linewidth = 0.5, alpha = a, zorder=2)
    else:
        hexdf_full.plot(ax=full_ax, column=prop, cmap=cmap, vmin=0, vmax=full_vmax, legend=False, edgecolor='lightgray', linewidth = 0.5, alpha = a, zorder=2)
    full_ax.set_xlim([-2250000, 2150000])
    full_ax.set_ylim([-1500000, 2600000])

    ak.geometry.boundary.plot(ax=coast_ax, color='black', zorder=1)
    if dev==True:
        hexdf_coast.plot(ax=coast_ax, column=prop, cmap=cmap, norm=coastdivnorm, legend = False, edgecolor='lightgray', linewidth = 0.5, alpha = a, zorder=2)
    else:
        hexdf_coast.plot(ax=coast_ax, column=prop, cmap=cmap, vmin=0, vmax=coast_vmax, legend = False, edgecolor='lightgray', linewidth = 0.5, alpha = a, zorder=2)
    coast_ax.set_xlim([-2250000, 2150000])
    coast_ax.set_ylim([0, 2600000])

    #formatting axes text
    heatmap_ax.set(xlabel=None, ylabel=None)
    heatmap_ax.xaxis.tick_top()
    heatmap_ax.xaxis.set_tick_params(labeltop=True, size=0)
    heatmap_ax.xaxis.set_tick_params(labelbottom=False)
    heatmap_ax.yaxis.set_tick_params(size=0)
    heatmap_ax.set_xticklabels(labels=[str(y) for y in pt.columns.to_list()], rotation=45, fontsize=f)
    heatmap_ax.set_yticklabels(['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'], rotation=0, fontsize=f, ha='right')
    #heatmap_ax.set_yticklabels([str(m) for m in pt.index.to_list()],rotation=0, fontsize=f, ha='center')#option to change to numeric months

    plt.setp(full_ax, yticks=[], xticks=[])
    plt.setp(coast_ax, yticks=[], xticks=[])

    #color bar
    colorbar = fig.colorbar(heatmap_ax.collections[0], location='top', ticks=[pt_vmin, pt_vmax], shrink=.35, pad=0.18)
    colorbar.set_ticklabels([cbar_labels[0], cbar_labels[1]], fontsize=f-1, fontstyle='italic')
    colorbar.outline.set_linewidth(0)
    colorbar.ax.tick_params(size=0)

    #titles
    full_ax.set_title('Event Centers', y=-0.1, fontsize=f+1)
    coast_ax.set_title('Coastal Impact Points', y=-0.1, fontsize=f+1)
    fig.suptitle(title, y=.895, fontsize=f+3)

    #enso legend
    if type(phenom) != str:
        plt.figtext(.1, .9, phenom_string, fontsize=f-1)
    else:
        pass

    plt.show()


def events_prop_mmm(events):
    """Plots a six-panel min/mean/max for AR event properties. Original event attribute field names are hard-coded, so the events gdf input should not be altered before use in this function.

    Parameters
    ----------
    events : geodataframe
        geodataframe of events layer with all original attribute field names

    Returns
    -------
    A 6-panel min/mean/max figure.
    """
    #subset attributes, including time for x-axis
    #define easier-to-read titles (excluding time attribute)
    sub = events[['start', 'dur_hrs', 'tintensity', 'rintensity', 'ratio_m', 'len_km_m', 'dircoher_m']].copy()
    titles = ['duration (hrs)', 'total intensity', 'relative intensity', 'length/width ratio', 'length (km)', 'directional coherence (%)']

    #setup plot and axes
    fig, axes = plt.subplots(int(len(sub.drop(columns='start').columns)/2), 2, sharex=False, figsize=(8, 7))
    plt.subplots_adjust(hspace = 0.6)
    sns.set_style("white")
    sns.despine(fig=fig, ax=axes, top=True, right=True, left=True, bottom=True, offset=None, trim=False)

    #define color pairs....must be the same # of pairs as # of non-time attributes
    colors = ['green', 'teal', 'firebrick', 'olive', 'mediumorchid', 'navy']
    #set fontsize for attr text
    f = 8

    #loop thru attributes and plot
    for attr, ax, t, c in zip(sub.drop(columns='start').columns.values.tolist(), fig.axes, titles, colors):
        #get stats
        a_min, a_mean, a_max = sub[attr].values.min(), sub[attr].values.mean(), sub[attr].values.max()
        #get time values for each stat
        a_min_idx, a_max_idx = sub[attr].idxmin(), sub[attr].idxmax()
        min_t, max_t = sub['start'].iloc[sub[attr].idxmin()], sub['start'].iloc[sub[attr].idxmax()]
        #mean time value will just be the midpoint between min and max
        mean_t = min_t + (max_t - min_t) / 2 
        #set time values for mean line extension; use 25% of distance between min_t and max_t
        mean_t_low = min_t + (max_t - min_t) / 4
        mean_t_high = max_t - (max_t - min_t) / 4
        #plot data cloud and mean line extension
        sns.scatterplot(ax=ax, data=sub, x='start', y=attr, color='lightgray', s=2.75)
        sns.lineplot(ax=ax, x=[mean_t_low, mean_t, mean_t_high], y=[a_mean, a_mean, a_mean], color=c, linewidth=.6, linestyle=(0, (3,3)))
        #plot min/mean/max points
        ax.plot(min_t, a_min, marker='o', fillstyle='none', markeredgewidth= .5, c=c)
        ax.plot(max_t, a_max, marker='o', fillstyle='full', markeredgewidth= .5, c=c)
        ax.plot(mean_t, a_mean, marker='o', fillstyle='bottom', markeredgewidth= .5, c=c)
        #define relative nudge distance for point labels (divides the y-axis into 50 parts)
        nudge = (a_max - a_min)/50
        #add point labels, with nudge for formatting
        ax.text(x = min_t, y = a_min + nudge, s = str("  " + str(int(a_min))), color = c, fontsize=f)
        ax.text(x = max_t, y = a_max - nudge*3, s = str("  " + str(int(a_max))), color = c, fontsize=f)
        ax.text(x = mean_t, y = a_mean + nudge, s = str("  " + str(int(a_mean))), color = c, fontsize=f)
        #adjust axis labels
        ax.set_ylabel(t, rotation=0, labelpad=-105, color=c, fontsize=f+2, y=-0.15)
        ax.set_xlabel('t  >', rotation=0, labelpad=3, color='gray', fontsize=f-1, style='italic', x=0.1)
    #remove axes ticks and add title    
    plt.setp(axes, yticks=[], xticks=[])
    fig.suptitle(('AR event properties: min, mean, max (' + str(start_year) + '-' + str(end_year) + ')\nn = ' + str(len(events))), color='black', fontsize=f+4, y=.95)

    plt.show()


def events_pairplot(events):
    """Plots a pairplot of AR event properties. Original event attribute field names are hard-coded, so the events gdf input should not be altered before use in this function.

    Parameters
    ----------
    events : geodataframe
        geodataframe of events layer with all original attribute field names

    Returns
    -------
    A pairplot figure.
    """
    sub = events[['dur_hrs', 'rintensity', 'tintensity', 'ratio_m', 'len_km_m', 'dircoher_m']].copy()
    sub.rename(columns={'dur_hrs':'duration (hrs)', 'rintensity':'relative intensity', 'tintensity':'total intensity', 'ratio_m':'length/width ratio', 'len_km_m':'length (km)', 'dircoher_m':'mean directional coherence (%)'}, inplace=True)
    sns.set_style("white")

    g = sns.PairGrid(sub)
    g.map_upper(sns.scatterplot, size=0.1, color='mediumpurple')
    g.map_lower(sns.kdeplot, fill=True)
    g.map_diag(sns.histplot, kde=True, color='orange')

    g.fig.suptitle('Pairplot of AR Event Properties', y=1.01, fontsize=16)

    plt.show()


def convert_df_to_plus_minus_array(input_df, low, high):
    """Converts a dataframe to a numpy array, replacing values below or above the low/high thresholds with +/- strings, and values between low/high thresholds with an empty value ('').

    Parameters
    ----------
    input_df : dataframe
        dataframe of floats or integers
    low : float or integer
        low value threshold
    high : float or integer
        high value threshold

    Returns
    -------
    Array of +/-/empty values.
    """
    df = input_df.copy()
    for col in df.columns:
        df.loc[df[col] >= high, col] = 99
        df.loc[df[col] <= low, col] = -99
        df.loc[(low < df[col]) & (df[col] < high), col] = 555
    df.replace({99:'+', -99:'-', 555:''}, inplace=True)
    a = df.to_numpy()
    return a