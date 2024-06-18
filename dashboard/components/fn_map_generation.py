import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import plotly.graph_objects as go
import plotly.express as px
import json
import h3pandas

colourscale = [((0.0, i), (1.0, i)) for i in px.colors.qualitative.Pastel]

def create_grid(grid_size):
    """
    Creates a grid of rectangles over the given area
    @param grid_size:
    @return: indexed grid
    """
    xmin, ymin, xmax, ymax = 17.94, 59.28, 18.18, 59.39
    cell_size = grid_size  # 0.002
    cell_sizex = cell_size * 2
    cell_sizey = cell_size

    # create the cells in a loop
    grid_cells = []
    for x0 in np.arange(xmin, xmax + cell_sizex, cell_sizex):
        for y0 in np.arange(ymin, ymax + cell_sizey, cell_sizey):
            # bounds
            x1 = x0 - cell_sizex
            y1 = y0 + cell_sizey
            grid_cells.append(shapely.geometry.box(x0, y0, x1, y1))
    cell = gpd.GeoDataFrame(grid_cells, columns=['geometry'])
    return cell


def create_aggregated_zones(gdf, drop_val, radio_aggr, grid_choice):
    """
    Merges rectangle grid with aggregated values for each zone
    @param grid_choice: string (sqr for square, hexa for hexagonal)
    @param radio_aggr: string, aggregation mode choice
    @param gdf: geo-dataframe
    @param drop_val: selected variable
    @return: grid with aggregated values
    """
    if grid_choice == 'sqr':
        # Square Grid
        cell = create_grid(0.002)
        merged = gpd.sjoin(gdf, cell, how='left')

        az_aggr = pd.DataFrame()
        az_aggr['demand_total'] = merged.groupby('index_right').count()[['id']]

        az_aggr[drop_val] = variable_aggregation_function(merged, drop_val, radio_aggr)
        az_aggr = az_aggr[az_aggr['demand_total'] > 10]
        # merge zones with grid coordinates
        for j in az_aggr.columns:
            cell.loc[az_aggr.index, j] = az_aggr[j].values

        cell.set_crs(epsg="4326", inplace=True)
        return cell

    else:
        # Hex Grid
        res = 9
        merged = gdf.dropna().h3.geo_to_h3(res, lat_col='o_lat', lng_col='o_lng')
        rescol = 'h3_0' + str(res) if res < 10 else 'h3_' + str(res)
        merged = merged.reset_index()
        merged = merged.rename(columns={rescol: 'index_right'})

        az_aggr = pd.DataFrame()
        az_aggr['demand_total'] = merged.groupby('index_right').count()[['id']]

        az_aggr[drop_val] = variable_aggregation_function(merged, drop_val, radio_aggr)
        az_aggr = az_aggr[az_aggr['demand_total'] > 10]

        az_aggr = az_aggr.h3.h3_to_geo_boundary()
        return az_aggr

def get_zoom_center(longitudes=None, latitudes=None):
    """
    Determine zoom and center coordiantes based on the data for the map
    Basic framework adopted from Krichardson and jcmontalbano under the following thread:
    https://community.plotly.com/t/dynamic-zoom-for-mapbox/32658/7
    @param longitudes: data series
    @param latitudes: data series
    @return: zoom, center coordinates (lng, lat)
    """
    if len(latitudes) != len(longitudes):
        return 0, [0, 0]

    # Get the boundary-box
    b_box = {'height': latitudes.max() - latitudes.min(),
             'width': longitudes.max() - longitudes.min(),
             'center': [np.mean(longitudes), np.mean(latitudes)]}

    # get the area of the bounding box in order to calculate a zoom-level
    area = b_box['height'] * b_box['width']

    # * 1D-linear interpolation
    zoom = np.interp(x=area,
                     xp=[0, 5 ** -10, 4 ** -10, 3 ** -10, 2 ** -10, 1 ** -10, 1 ** -5],
                     fp=[20, 15, 14, 13, 12, 7, 5])

    return int(zoom), b_box['center']


def generate_single_map(zone_aggr, sel_var):
    """
    Generates folium map for a single timeframe
    @param zone_aggr: grid with aggregated values
    @param sel_var: selected variable
    @return: rendered HTMl variable
    """
    diffcolor = 'YlOrBr_r' if sel_var == 'GHG' else 'YlGn' if sel_var in ['speed', 'reduced_time'] else 'YlOrBr'
    zoom, center = get_zoom_center(zone_aggr.geometry.centroid.x, zone_aggr.geometry.centroid.y)

    m = folium.Map(location=(center[1], center[0]), zoom_start=zoom, tiles='CartoDB Positron')
    folium.TileLayer('cartodbpositron').add_to(m)
    folium.TileLayer('openstreetmap').add_to(m)
    folium.TileLayer('cartodbdark_matter').add_to(m)

    folium.Choropleth(zone_aggr, data=zone_aggr,
                      columns=['id', sel_var],
                      key_on='feature.properties.id',
                      fill_color=diffcolor,
                      nan_fill_opacity=0,
                      line_opacity=0,
                      legend_name=sel_var,
                      smooth_factor=0,
                      Highlight=True,
                      name=sel_var,
                      bins=8
                      ).add_to(m)

    folium.LayerControl().add_to(m)

    style_function = lambda x: {'fillColor': diffcolor,
                                'color': '#000000',
                                'fillOpacity': 0,
                                'weight': 0}
    highlight_function = lambda x: {'fillColor': '#000000',
                                    'color': '#000000',
                                    'fillOpacity': 0.50,
                                    'weight': 0.1}
    nil = folium.features.GeoJson(
        data=zone_aggr,
        style_function=style_function,
        control=False,
        highlight_function=highlight_function,
        tooltip=folium.features.GeoJsonTooltip(
            fields=['id', sel_var],
            aliases=['id', sel_var],
            style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
        )
    )

    m.add_child(nil)
    m.keep_in_front(nil)

    src_doc = m.get_root().render()
    return src_doc


def generate_compare_map(zone_bef, zone_aft, sel_var):
    """
    Generates folium map for a single timeframe
    @param zone_bef: grid with aggreated values, before
    @param zone_aft: grid with aggregated values, after
    @param sel_var: selected variable
    @return: rendered HTMl variable
    """
    diffcolor = 'RdYlGn' if sel_var != 'reduced_time' else 'RdYlGn_r'

    cell_diff = zone_aft.copy()
    cell_diff[sel_var] = cell_diff[sel_var] - zone_bef[sel_var]
    cell_diff = cell_diff[cell_diff[sel_var].notna()]
    cell_diff[sel_var] = cell_diff[sel_var].round(2)

    scale_max = max(abs(min(cell_diff[sel_var])), abs(abs(max(cell_diff[sel_var]))))
    bins = np.linspace(-scale_max, scale_max, 9)

    zoom, center = get_zoom_center(cell_diff.geometry.centroid.x, cell_diff.geometry.centroid.y)

    m = folium.Map(location=(center[1], center[0]), zoom_start=zoom, tiles='CartoDB Positron')
    folium.TileLayer('cartodbdark_matter').add_to(m)
    folium.TileLayer('cartodbpositron').add_to(m)
    folium.Choropleth(cell_diff, data=cell_diff,
                      columns=['id', sel_var],
                      key_on='feature.properties.id',
                      fill_color=diffcolor,
                      nan_fill_opacity=0,
                      line_opacity=0,
                      legend_name=sel_var,
                      smooth_factor=0,
                      Highlight=True,
                      name=sel_var,
                      bins=bins
                      ).add_to(m)

    folium.LayerControl().add_to(m)

    style_function = lambda x: {'fillColor': diffcolor,
                                'color': '#000000',
                                'fillOpacity': 0,
                                'weight': 0}
    highlight_function = lambda x: {'fillColor': '#000000',
                                    'color': '#000000',
                                    'fillOpacity': 0.50,
                                    'weight': 0.1}
    nil = folium.features.GeoJson(
        data=cell_diff,
        style_function=style_function,
        control=False,
        highlight_function=highlight_function,
        tooltip=folium.features.GeoJsonTooltip(
            fields=['id', sel_var],
            aliases=['id', sel_var],
            style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
        )
    )

    m.add_child(nil)
    m.keep_in_front(nil)

    src_doc = m.get_root().render()
    return src_doc


def variable_aggregation_function(merged, var, aggr_mode):
    """
    Defines the aggregation function for a specific variable
    @param merged: df with assigned zone ids
    @param var: selected variable
    @param aggr_mode: aggregation mode, from aggr-radio
    @return: df with applied aggregation function
    """

    if aggr_mode == 'demand_total':
        return merged.groupby('index_right').count()[['id']]
    if aggr_mode == 'demand_dailysumavg':
        return \
            merged.groupby(['index_right', 'Date'])['id'].count().dropna().reset_index().groupby('index_right').mean()[
                'id']

    if aggr_mode == 'nunique_id_total':
        return merged.groupby(['index_right'])['id'].nunique()
    if aggr_mode == 'nunique_id_dailysumavg':
        return merged.groupby(['index_right', 'Date'])['id'].nunique().dropna().reset_index().groupby(
            'index_right').mean()['id']

    if aggr_mode == 'utilisationrate_avg':
        return merged.groupby(['index_right', 'id'])['id'].count().dropna().groupby('index_right').mean()
    if aggr_mode == 'utilisationrate_dailysumavg':
        return merged.groupby(['index_right', 'Date']).id.count().dropna().reset_index().groupby(
            ['index_right']).mean().id

    elif '_avg' in aggr_mode:
        return merged.groupby('index_right')[var].mean()

    elif '_dailysumavg' in aggr_mode:
        return merged.groupby(['index_right', 'Date'])[var].sum().reset_index().groupby('index_right').mean()[var]

    elif '_total' in aggr_mode:
        return merged.groupby('index_right')[var].sum()


def generate_modal_subs_map(aggr_zone):
    """
    Creates map for modal substitution rate, i.e. most likely mode per zone
    @param aggr_zone: geodataframe with aggregated zones
    @return: mapbox chloropleth graph object
    """
    fig = go.Figure()
    for i, sub in enumerate(aggr_zone['modal_sub'].unique()):
        dfp = aggr_zone[aggr_zone['modal_sub'] == sub]
        fig.add_choroplethmapbox(geojson=json.loads(dfp.to_json()), locations=dfp['id'],
                                 featureidkey="properties.id",
                                 z=[i, ] * len(dfp),
                                 showlegend=True, name=sub,
                                 colorscale=colourscale[i], showscale=False, marker_opacity=0.3, marker_line_width=0,
                                 hoverinfo='name', hoverlabel={'namelength': -1})

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                      mapbox=dict(style="carto-positron", zoom=11,
                                  center={'lat': 59.33, 'lon': 18.10}))
    return fig
