import datetime as dt
import random
import h3
import h3pandas
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

ratio_card = 0.3
ratio_male = 0.5
ratio_lic = 463746 / 1030000
ratio_md = 0.3

car_ratio = 0.26
bike_ratio = 0.15
distance_walk_threshold = 300

asc_walk = 2.01
asc_bike = 0.71
asc_pt = 0
asc_car = -0.786

b_walk_distance = -9.18
b_bike_distance = -2.19
b_pt_time = -1.38
b_pt_cost = -0.0388
b_pt_inner = 0.51
b_pt_card = 3.18
b_car_time = -0.482
b_car_cost = -0.0309
b_car_male = 0.253
b_lic = 0.575
b_md = 1.77

fuelprice_car = 1.6
fuelprice_taxi = 17


def parse_data(df):
    """
    Parses and preprocesses the input DataFrame containing trip data.

    @param df: DataFrame containing trip data.
    @return: Preprocessed DataFrame.
    """
    df = df.rename(columns={'distance': 'escooter_distance', 'car_duration': 'car_time', 'carDistance': 'car_distance',
                            'walk_duration': 'walk_time', 'walkDistance': 'walk_distance',
                            'transit_total_duration': 'transit_totaltime', 'transit_walkTime': 'transit_walk_time',
                            'transit_Time': 'transit_time', 'transit_waitingTime': 'transit_waiting_time',
                            'transit_walkDistance': 'transit_walkdistance'})

    cols = ['id', 'o_time', 'd_time', 'o_lat', 'o_lng', 'd_lat', 'd_lng', 'escooter_distance', 'escooter_time',
            'transit_time', 'transit_totaltime', 'transit_walk_time', 'transit_waiting_time', 'transit_walkdistance',
            'transit_transitdistance', 'car_time', 'car_distance', 'walk_time', 'walk_distance', 'type', 'ratio_',
            'U_walk', 'U_bike', 'U_PT', 'U_car', 'U_taxi', 's', 'P_walk', 'P_bike', 'P_PT', 'P_car', 'P_taxi', 'GHG']

    cols_final = list(set(cols) & set(df.columns))
    df = df[cols_final]

    df['o_time'] = pd.to_datetime(df['o_time'], errors='coerce')
    df['d_time'] = pd.to_datetime(df['d_time'], errors='coerce')
    df = df.sort_values(by=["o_time"]).reset_index(drop=True)
    df['day'] = df['o_time'].dt.day_name()
    df['Month'] = df['o_time'].dt.month
    df['year'] = df['o_time'].dt.year
    df['Date'] = df['o_time'].dt.day
    df['hour'] = df['o_time'].dt.hour
    df['isodate'] = df['o_time'].dt.normalize()

    df['escooter_time'] = df.d_time - df.o_time
    df.escooter_time = df.escooter_time.dt.seconds
    df['speed'] = df.escooter_distance / df.escooter_time

    # Initial data source used "vio"
    df['type'] = df['type'].replace('vio', 'voi') if 'type' in df.columns else 'voi'
    df[['car_distance', 'transit_transitdistance']] = df[['car_distance', 'transit_transitdistance']].fillna(0)

    return df


def clean_data(df_range, city_coordinates):
    """
    Cleans the input DataFrame containing trip data based on specified criteria.
    Adapted from Omkar Parishwad (https://github.com/parishwadomkar).

    @param df_range: DataFrame containing trip data.
    @param city_coordinates: Tuple containing city coordinates in the format (north, south, east, west).
    @return: Cleaned DataFrame.
    """
    print('#Datapoints - total: ' + str(df_range.shape[0]))

    # Filter for area
    north, south, east, west = city_coordinates
    mask_lng = (df_range['o_lng'] > east) | (df_range['d_lng'] > east) | (df_range['o_lng'] < west) | (
            df_range['d_lng'] < west)
    mask_lat = (df_range['o_lat'] > north) | (df_range['d_lat'] > north) | (df_range['o_lat'] < south) | (
            df_range['d_lat'] < south)
    mask = mask_lng | mask_lat
    df_city = df_range[~mask].reset_index(drop=True)
    print('#Datapoints -  area filter: ' + str(df_city.shape[0]))

    # Filter for distance
    df_city = df_city[df_city['escooter_distance'] < 20000]
    df_city = df_city[df_city['escooter_distance'] > 50]
    print('#Datapoints - distance filter: ' + str(df_city.shape[0]))

    # Filter for trip duration and speed
    df_city = df_city[(df_city['escooter_time'] > 30) & (df_city['escooter_time'] < 3600)]
    df_city['speed'] = df_city['escooter_distance'] / df_city['escooter_time']
    df_city = df_city[(df_city['speed'] < 10) & (df_city['speed'] > (2 / 3.6))]
    print('#Datapoints - speed & time filter: ' + str(df_city.shape[0]))

    return df_city


def missing_data_hours(df, before_start_date, before_end_date, after_start_date, after_end_date):
    """
    Calculates the number of missing data hours per day within specified date ranges.

    @param df: DataFrame containing trip data.
    @param before_start_date: Start date of the "before" period.
    @param before_end_date: End date of the "before" period.
    @param after_start_date: Start date of the "after" period.
    @param after_end_date: End date of the "after" period.
    @return: DataFrame containing the number of missing data hours per day.
    """
    # get all dates within the range
    date_ranges = [[before_start_date, before_end_date], [after_start_date, after_end_date]]
    dates = []
    for i in date_ranges:
        temp = i[0]
        while temp <= i[1]:
            dates.append(temp.isoformat())
            temp += dt.timedelta(days=1)

    # count number of hours per day where no data was recorded
    missing = df.groupby(['isodate', 'hour']).size().reset_index()
    missing['date_hour'] = missing['isodate'].astype(str) + '-' + missing['hour'].astype(str)
    missing_hour = pd.DataFrame(columns=['date', 'hours'])

    for i in dates:
        curr_date = missing[missing['isodate'] == i]
        hours = 0
        for j in range(0, 24):
            if curr_date[curr_date['hour'] == j].empty:
                hours += 1
        missing_hour.loc[len(missing_hour.index)] = [i, hours]

    missing_hour['date'] = missing_hour['date'].astype('str')
    return missing_hour


def calculate_mode_likelihood_ghg(df):
    """
    Calculates the likelihood of different transport modes and greenhouse gas emissions for each trip.

    @param df: DataFrame containing trip data.
    @return: DataFrame with calculated mode likelihoods and greenhouse gas emissions.
    """

    df['ratio_'] = df.apply(lambda x: return_mode_availability(x["walk_distance"]), axis=1)

    print('Calculating utilities...')
    df['U_walk'] = df.apply(lambda x: calculate_utility_walkbike('walk', x['escooter_distance']), axis=1)
    df['U_bike'] = df.apply(lambda x: calculate_utility_walkbike('bike', x['escooter_distance']), axis=1)
    df['U_PT'] = df.apply(
        lambda x: calculate_utility_pt(x["transit_time"], x["transit_waiting_time"], run_monte_carlo(ratio_card)),
        axis=1)
    df['U_car'] = df.apply(
        lambda x: calculate_utility_car_taxi('car', x["car_time"], x["car_distance"], run_monte_carlo(ratio_lic),
                                             run_monte_carlo(ratio_male), run_monte_carlo(ratio_md)), axis=1)
    df['U_taxi'] = df.apply(lambda x: calculate_utility_car_taxi('taxi', x["car_time"], x["car_distance"], 0, 0, 0),
                            axis=1)

    df[['U_walk', 'U_bike', 'U_PT', 'U_car', 'U_taxi']] = df[['U_walk', 'U_bike', 'U_PT', 'U_car', 'U_taxi']].fillna(
        -np.inf)

    df['s'] = df.apply(
        lambda x: np.exp(x["U_walk"]) * x['ratio_'][0] + np.exp(x["U_bike"]) * x['ratio_'][1] + np.exp(x['U_PT']) *
                  x['ratio_'][2] + np.exp(x['U_car']) * x['ratio_'][3] + np.exp(x['U_taxi']), axis=1)

    print('Calculating probabilities...')
    df['P_walk'] = df.apply(lambda x: np.exp(x["U_walk"]) * x['ratio_'][0] / x['s'], axis=1)
    df['P_bike'] = df.apply(lambda x: np.exp(x["U_bike"]) * x['ratio_'][1] / x['s'], axis=1)
    df['P_PT'] = df.apply(lambda x: np.exp(x["U_PT"]) * x['ratio_'][2] / x['s'], axis=1)
    df['P_car'] = df.apply(lambda x: np.exp(x["U_car"]) * x['ratio_'][3] / x['s'], axis=1)
    df['P_taxi'] = df.apply(lambda x: np.exp(x["U_taxi"]) / x['s'], axis=1)

    print('Calculating GHG and reduced time...')
    df['GHG'] = df.car_distance / 1000 * 160.7 * (
                df.P_car + df.P_taxi) + df.transit_transitdistance / 1000 * 16.04 * df.P_PT + df.escooter_distance / 1000 * 37.0 * df.P_bike - df.escooter_distance / 1000 * 67

    df['reduced_time'] = ((df['P_walk'] * df['walk_time']) + (df['P_bike'] * df['escooter_time']) + (
            df['P_PT'] * df['transit_totaltime']) + (df['P_car'] * df['car_time']) + (df['P_taxi'] * df['car_time'])) - \
                         df['escooter_time']

    return df


def run_monte_carlo(p):
    """
    Run monte carlo simulation for p.
    Code from Ruo Jia (https://research.chalmers.se/en/person/ruoj)
    @param p: float
    @return:
    """
    result = 1 if random.random() < p else 0
    return result


def return_mode_availability(distance):
    """
    Returns availability likelihood of transport modes.
    Code from Ruo Jia (https://research.chalmers.se/en/person/ruoj)

    @param distance: float, trip distance
    @return: probability ratio for [ratio_walk, ratio_bike, ratio_pt, ratio_car]
    """
    if distance < distance_walk_threshold:
        # below a certain distance only walking is assumed
        return 1, 0, 0, 0
    else:
        # simulate ownership of car & bike
        return 1, run_monte_carlo(bike_ratio), 1, run_monte_carlo(car_ratio)


def calculate_utility_walkbike(mode, distance):
    """
    Returns the utility of walking and biking.
    Code from Ruo Jia (https://research.chalmers.se/en/person/ruoj)

    @param mode: string, walk or bike
    @param distance: float, distance taken by mode
    @return: float, utility
    """
    distance_km = distance / 1000
    if mode == 'walk':
        return asc_walk + b_walk_distance * distance_km
    else:
        return asc_bike + b_walk_distance * distance_km


def calculate_utility_pt(transit_time, waiting_time, mc):
    """
    Returns the utility of public transport.
    Code from Ruo Jia (https://research.chalmers.se/en/person/ruoj)

    @param transit_time: float, Total time taken by public transport
    @param waiting_time: float, Waiting time in between connections
    @param mc: float, Probability of owning PT card/abonnement
    @return: float, utility
    """
    trip_cost = 36
    return asc_pt + b_pt_time * (
            (transit_time / 3600) + (waiting_time / 3600)) + b_pt_cost * trip_cost + b_pt_inner + b_pt_card * mc


def calculate_utility_car_taxi(mode, car_time, car_distance, mc_lc, mc_male, mc_md):
    """
    Returns the utility of going by car or taxi
    Code from Ruo Jia (https://research.chalmers.se/en/person/ruoj)

    @param mode: string, car or taxi
    @param car_time: float, duration taken by mode
    @param car_distance: float, distance taken by mode
    @param mc_lc: float, Probability of owning a license (only for car)
    @param mc_male: float, Probability of male driver (only for car)
    @param mc_md: float, Probability of ?? (only for car)
    @return: float, utility
    """
    v_car = asc_car + b_car_time * car_time
    if mode == 'taxi':
        return v_car + b_car_cost * (61 + car_distance / 1000 * fuelprice_taxi)
    else:
        return v_car + b_car_cost * (
                car_distance / 1000 * fuelprice_car) + b_lic * mc_lc + b_car_male * mc_male + b_md * mc_md


def create_zones(df, grid_choice, grid_size, city_coordinates, min_data, base):
    """
    Creates zones based on the selected grid choice

    @param df: DataFrame
    @param grid_choice: string, choice of grid shape ('square', 'hexagon', or 'shapefile')
    @param grid_size: float or str, size of the grid or path to shapefile
    @param city_coordinates: tuple, coordinates of the city (latitude, longitude)
    @param min_data: int, minimum data threshold
    @param base: string, base name for columns (e.g., 'o' for origin or 'd' for destination)

    @return: GeoDataFrame with aggregated zones
    """
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[base + '_lng'], df[base + '_lat']))

    if grid_choice == 'square':
        zone_aggr = create_square_grid(gdf, grid_size, city_coordinates, min_data)
    elif grid_choice == 'hexagon':
        zone_aggr = create_hexagonal_grid(gdf, grid_size, base, min_data)
    elif grid_choice == 'shapefile':
        # grid_size is being used as proxy for path to shapefile
        zone_aggr = create_shapefile_grid(gdf, grid_size, min_data)
    else:
        print('Invalid grid shape')
        return None

    zone_aggr = zone_aggr[zone_aggr['demand_total'] > 0]
    zone_aggr['id'] = zone_aggr.index
    return zone_aggr


def create_square_grid(gdf, grid_size, city_coordinates, min_data):
    """
    Creates square grid cells based on specified cell size and aggregates data within each cell

    @param gdf: GeoDataFrame, input data containing geometry information
    @param grid_size: float, size of the grid cell
    @param city_coordinates: tuple, coordinates of the city (ymax, ymin, xmax, xmin)
    @param min_data: int, minimum data threshold for aggregation

    @return: GeoDataFrame with aggregated zones
    """
    ymax, ymin, xmax, xmin = city_coordinates
    cell_size = grid_size
    cell_sizex = cell_size * 2
    cell_sizey = cell_size

    # create square cells
    grid_cells = []
    for x0 in np.arange(xmin, xmax + cell_sizex, cell_sizex):
        for y0 in np.arange(ymin, ymax + cell_sizey, cell_sizey):
            x1 = x0 - cell_sizex
            y1 = y0 + cell_sizey
            grid_cells.append(shapely.geometry.box(x0, y0, x1, y1))
    cell = gpd.GeoDataFrame(grid_cells, columns=['geometry'])

    merged = gpd.sjoin(gdf, cell, how='left')
    az_aggr = aggregate_zonewise(merged, min_data)

    # merge zones with grid coordinates
    for j in az_aggr.columns:
        cell.loc[az_aggr.index, j] = az_aggr[j].values

    cell.set_crs(epsg="4326", inplace=True)
    return cell


def create_hexagonal_grid(gdf, res, base, min_data):
    """
    Creates hexagonal grid cells based on specified resolution and aggregates data within each cell

    @param gdf: GeoDataFrame, input data containing geometry information
    @param res: int, resolution of the hexagonal grid
    @param base: string, prefix for latitude and longitude columns
    @param min_data: int, minimum data threshold for aggregation

    @return: GeoDataFrame with aggregated zones
    """
    merged = gdf.dropna().h3.geo_to_h3(res, lat_col=base + '_lat', lng_col=base + '_lng')
    rescol = 'h3_0' + str(res) if res < 10 else 'h3_' + str(res)

    merged = merged.reset_index()
    merged = merged.rename(columns={rescol: 'index_right'})
    az_aggr = aggregate_zonewise(merged, min_data)

    az_aggr = az_aggr.h3.h3_to_geo_boundary()
    return az_aggr .set_crs(epsg="4326", inplace=True, allow_override=True)


def create_shapefile_grid(gdf, path, min_data):
    """
    Creates grid cells based on a shapefile and aggregates data within each cell

    @param gdf: GeoDataFrame, input data containing geometry information
    @param path: string, path to the shapefile
    @param min_data: int, minimum data threshold for aggregation

    @return: GeoDataFrame with aggregated zones
    """
    shpfile = gpd.read_file(path)
    shpfile = shpfile.to_crs(epsg=4326)

    merged = gpd.sjoin(gdf, shpfile, how='left')
    merged = merged.rename(columns={'id_left': 'id'})
    az_aggr = aggregate_zonewise(merged, min_data)

    for j in az_aggr.columns:
        shpfile.loc[az_aggr.index, j] = az_aggr[j].values

    shpfile.set_crs(epsg="4326", inplace=True)
    return shpfile


def aggregate_zonewise(merged, min_data):
    """
    Aggregates variables in different dimensions within each zone based on the merged DataFrame

    @param merged: DataFrame containing merged data
    @param min_data: Minimum number of trips required for a zone to be considered
    @return: GeoDataFrame containing zone-wise attributes
    """
    print('Aggregating zones...')
    az_aggr = pd.DataFrame()
    az_aggr['demand_total'] = merged.groupby('index_right').count()[['id']]
    az_aggr['demand_dailysumavg'] = \
        merged.groupby(['index_right', 'Date'])['id'].count().dropna().reset_index().groupby('index_right').mean()['id']

    az_aggr['escooter_distance_avg'] = merged.groupby('index_right')['escooter_distance'].mean()
    az_aggr['escooter_time_avg'] = merged.groupby('index_right')['escooter_time'].mean()
    az_aggr['speed_avg'] = merged.groupby('index_right')['speed'].mean()

    az_aggr['GHG_total'] = merged.groupby('index_right')['GHG'].sum()
    az_aggr['GHG_avg'] = merged.groupby('index_right')['GHG'].mean()
    az_aggr['GHG_dailysumavg'] = \
        merged.groupby(['index_right', 'Date'])['GHG'].sum().reset_index().groupby('index_right').mean()['GHG']

    az_aggr['P_walk_avg'] = merged.groupby('index_right')['P_walk'].mean()
    az_aggr['P_bike_avg'] = merged.groupby('index_right')['P_bike'].mean()
    az_aggr['P_PT_avg'] = merged.groupby('index_right')['P_PT'].mean()
    az_aggr['P_car_avg'] = merged.groupby('index_right')['P_car'].mean()
    az_aggr['P_taxi_avg'] = merged.groupby('index_right')['P_taxi'].mean()

    # 'walk_time', 'car_time','transit_totaltime','transit_time', 'transit_walktime'
    az_aggr['T_walk_avg'] = merged.groupby('index_right')['walk_time'].mean()
    az_aggr['T_car_avg'] = merged.groupby('index_right')['car_time'].mean()
    az_aggr['T_transittotal_avg'] = merged.groupby('index_right')['transit_totaltime'].mean()
    az_aggr['T_transitPT_avg'] = merged.groupby('index_right')['transit_time'].mean()
    az_aggr['T_transitwalk_avg'] = merged.groupby('index_right')['transit_walk_time'].mean()

    az_aggr['reduced_time_total'] = merged.groupby('index_right')['reduced_time'].sum()
    az_aggr['reduced_time_avg'] = merged.groupby('index_right')['reduced_time'].mean()
    az_aggr['reduced_time_dailysumavg'] = \
        merged.groupby(['index_right', 'Date'])['reduced_time'].sum().reset_index().groupby('index_right').mean()[
            'reduced_time']

    az_aggr['alternative_time_avg'] = az_aggr['reduced_time_avg'] + az_aggr['escooter_time_avg']

    az_aggr['utilisationrate_avg'] = merged.groupby(['index_right', 'id'])['id'].count().dropna().groupby(
        'index_right').mean()  # average utilisation rate
    az_aggr['nunique_id_total'] = merged.groupby(['index_right'])['id'].nunique()  # number of unique scooters
    az_aggr['nunique_id_dailysumavg'] = \
        merged.groupby(['index_right', 'Date'])['id'].nunique().dropna().reset_index().groupby('index_right').mean()[
            'id']

    # filter out zones with less than 30 trips in total
    az_aggr = az_aggr[az_aggr['demand_total'] > min_data]
    print(az_aggr.shape)
    return az_aggr
