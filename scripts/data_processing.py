import pandas as pd


def parse_data(df):
    df = df.rename(columns={'distance': 'escooter_distance', 'transit_Time': 'transit_time',
                            'transit_total_duration': 'transit_totaltime', 'transit_walkTime': 'transit_walkdistance',
                            'car_duration': 'car_time', 'carDistance': 'car_distance', 'walk_duration': 'walk_time',
                            'walkDistance': 'walk_distance', 'walkTime_y': 'walk_time',
                            'walkDistance_y': 'walk_distance'})

    cols = ['id', 'o_time', 'd_time', 'o_lat', 'o_lng', 'd_lat', 'd_lng',
            'escooter_distance', 'escooter_time', 'transit_time',
            'transit_totaltime', 'transit_walkdistance', 'transit_transitdistance',
            'car_time', 'car_distance', 'walk_time', 'walk_distance', 'type',
            'ratio_', 'U_walk', 'U_bike', 'U_PT', 'U_car', 'U_taxi', 's', 'P_walk',
            'P_bike', 'P_PT', 'P_car', 'P_taxi', 'GHG']

    cols_final = list(set(cols) & set(df.columns))
    df = df[cols_final]

    df['o_time'] = pd.to_datetime(df['o_time'], errors='coerce')
    df['d_time'] = pd.to_datetime(df['d_time'], errors='coerce')
    df = df.sort_values(by=["o_time"]).reset_index(drop=True)
    df['day'] = df['o_time'].dt.day_name()
    df['Month'] = df['o_time'].dt.month
    df['year'] = df['o_time'].dt.year
    df['Date'] = df['o_time'].dt.day
    df['isodate'] = df['o_time'].dt.normalize()

    df['escooter_time'] = df.d_time - df.o_time
    df.escooter_time = df.escooter_time.dt.seconds
    df['speed'] = df.escooter_distance / df.escooter_time

    # Initial data source used "vio"
    df['type'] = df['type'].replace('vio', 'voi') if 'type' in df.columns else 'voi'

    return df


def clean_data(df_range, city_coordinates):
    """
    Adapted cleaning from Omkar Parishwad
    :param df_range:
    :return:
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


def missing_data_hours(df, maxdays):
    missing = df.groupby(['isodate', 'hour']).size().reset_index()
    missing['date_hour'] = missing['isodate'].astype(str) + '-' + missing['hour'].astype(str)

    # change to datetime range first to last day in month/timeframes
    for i in range(1, maxdays + 1):
        curr_date = missing[missing['Date'] == i]
        for j in range(0, 24):
            if curr_date[curr_date['hour'] == j].empty:
                list_row = [i, j, 0, str(i) + '-' + str(j)]
                missing.loc[len(missing)] = list_row

    missing = missing[missing[0] < 1].sort_values(by=['date_hour']).groupby(
        'Date').size().to_frame().reset_index().rename(columns={0: '#Hours without data'})
    return missing


def enrich_data(df):

    df['transit_waiting_time'] = df['transit_totaltime'] - df['transit_walk_time'] - df['transit_time']

    df['reduced_time'] = ((df['P_walk'] * df['walk_time']) + (
            df['P_bike'] * df['escooter_time']) + (df['P_PT'] * df['transit_totaltime']) + (
                                  df['P_car'] * df['car_time']) + (
                                  df['P_taxi'] * df['car_time'])) - df['escooter_time']

    # add GHG, U_x etc
    return df

