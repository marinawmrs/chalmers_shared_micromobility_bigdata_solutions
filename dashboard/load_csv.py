import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from components.fn_preprocessing import availability, U_walk, U_bike, U_PT, run_monte_carlo, U_car, U_taxi

base_dir = os.path.abspath(os.path.dirname(__file__))
database_path = os.path.join(base_dir, 'tmp', 'trips.db')
os.makedirs(os.path.dirname(database_path), exist_ok=True)

engine = create_engine(f'sqlite:///{database_path}')

csv_file_path = 'components/data/rawraw_data.csv'
df = pd.read_csv(csv_file_path)


def parse_data(df):
    """
    Processes dataframe and calculates modal substitution and GHG emissions per trip
    @param df: 'raw' dataframe
    @return: 'processed' dataframe
    """
    df = df[
        ['id', 'isodate', 'o_time', 'd_time', 'o_lat', 'o_lng', 'd_lat', 'd_lng', 'distance', 'type', 'car_duration',
         'carDistance', 'walk_duration', 'walkDistance', 'transit_total_duration', 'transit_walkTime', 'transit_Time',
         'transit_waitingTime', 'transit_walkDistance', 'transit_transitdistance']]

    df.o_time = pd.to_datetime(df.o_time)
    df.d_time = pd.to_datetime(df.d_time)

    df['hour'] = df['o_time'].dt.hour
    df['day'] = df['o_time'].dt.day_name()
    df['Date'] = df['o_time'].dt.day
    df['escooter_time'] = df.d_time - df.o_time
    df.escooter_time = df.escooter_time.dt.seconds
    df = df.rename(columns={'distance': 'escooter_distance', 'car_duration': 'car_time', 'carDistance': 'car_distance',
                            'walk_duration': 'walk_time', 'walkDistance': 'walk_distance',
                            'transit_total_duration': 'transit_totaltime', 'transit_walkTime': 'transit_walk_time',
                            'transit_Time': 'transit_time', 'transit_waitingTime': 'transit_waiting_time',
                            'transit_walkDistance': 'transit_walkdistance'})

    df['speed'] = df.escooter_distance / df.escooter_time
    df = df[(df.speed < 10)]

    df['transit_waiting_time'] = df['transit_totaltime'] - df['transit_walk_time'] - df['transit_time']
    df[['car_distance', 'transit_transitdistance']] = df[['car_distance', 'transit_transitdistance']].fillna(0)

    df['ratio_'] = df.apply(lambda x: availability(x["walk_distance"]), axis=1)

    ratio_card = 0.3
    ratio_male = 0.5
    ratio_lic = 463746 / 1030000

    df['U_walk'] = df['walk_distance'].apply(U_walk)
    df['U_bike'] = df.apply(lambda x: U_bike(x["escooter_distance"]), axis=1)
    df['U_PT'] = df.apply(lambda x: U_PT(x["transit_time"], x["transit_waiting_time"], run_monte_carlo(ratio_card)),
                          axis=1)
    df['U_car'] = df.apply(
        lambda x: U_car(x["car_time"], x["car_distance"], run_monte_carlo(ratio_lic), run_monte_carlo(ratio_male),
                        run_monte_carlo(0.3)), axis=1)
    df['U_taxi'] = df.apply(lambda x: U_taxi(x["car_time"], x["car_distance"], run_monte_carlo(ratio_male)), axis=1)

    df[['U_walk', 'U_bike', 'U_PT', 'U_car', 'U_taxi']] = df[['U_walk', 'U_bike', 'U_PT', 'U_car', 'U_taxi']].fillna(
        -np.inf)

    df['s'] = df.apply(
        lambda x: np.exp(x["U_walk"]) * x['ratio_'][0] + np.exp(x["U_bike"]) * x['ratio_'][1] + np.exp(x['U_PT']) *
                  x['ratio_'][2] + np.exp(x['U_car']) * x['ratio_'][3] + np.exp(x['U_taxi']), axis=1)

    df['P_walk'] = df.apply(lambda x: np.exp(x["U_walk"]) * x['ratio_'][0] / x['s'], axis=1)
    df['P_bike'] = df.apply(lambda x: np.exp(x["U_bike"]) * x['ratio_'][1] / x['s'], axis=1)
    df['P_PT'] = df.apply(lambda x: np.exp(x["U_PT"]) * x['ratio_'][2] / x['s'], axis=1)
    df['P_car'] = df.apply(lambda x: np.exp(x["U_car"]) * x['ratio_'][3] / x['s'], axis=1)
    df['P_taxi'] = df.apply(lambda x: np.exp(x["U_taxi"]) / x['s'], axis=1)

    df['GHG'] = df.car_distance / 1000 * 160.7 * (
            df.P_car + df.P_taxi) + df.transit_transitdistance / 1000 * 16.04 * df.P_PT + df.escooter_distance / 1000 * 37.0 * df.P_bike - df.escooter_distance / 1000 * 67

    df['reduced_time'] = ((df['P_walk'] * df['walk_time']) + (df['P_bike'] * df['escooter_time']) + (
            df['P_PT'] * df['transit_totaltime']) + (df['P_car'] * df['car_time']) + (df['P_taxi'] * df['car_time'])) - \
                         df['escooter_time']

    df = df.drop(columns=['ratio_'])
    return df


df = parse_data(df)
print(df.head())
df.to_sql('trip', con=engine, if_exists='replace', index=False)
print("CSV data loaded.")
