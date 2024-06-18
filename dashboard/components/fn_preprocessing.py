import random
import os
import pandas as pd
from sqlalchemy import create_engine, text

pd.options.mode.copy_on_write = True

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
database_path = os.path.join(base_dir, 'tmp', 'trips.db')
engine = create_engine(f'sqlite:///{database_path}')


def query_data_within_range(operators, start_date, end_date, fields, days):
    """
    Query data from hypothetical database
    @param operators: array of operators to filter for
    @param start_date: iso-date string
    @param end_date: iso-date string
    @param fields: list of fields to select in the query.
    @param days: list of days of the week to include
    @return: dataframe
    """

    ops = ",".join(f"'{operator}'" for operator in operators)
    fieldstr = ", ".join(fields)
    daystr = ", ".join(f"'{day}'" for day in days)
    query = text(f"""
            SELECT {fieldstr} 
            FROM trip
            WHERE isodate BETWEEN :start_date AND :end_date
            AND type IN ({ops})
            AND day IN ({daystr})
        """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={'start_date': start_date, 'end_date': end_date})

    if 'o_time' in fields:
        df.o_time = pd.to_datetime(df.o_time)

    return df


def get_day_list(radio_var):
    day_mapping = {
        1: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        2: ['Monday', 'Tuesday', 'Wednesday', 'Thursday'],
        3: ['Friday'],
        4: ['Saturday', 'Sunday']
    }
    return day_mapping.get(radio_var, [])


def run_monte_carlo(p):
    """
    Run monte carlo simulation for p
    Code from Ruo Jia (https://research.chalmers.se/en/person/ruoj)
    @param p: float
    @return:
    """
    result = 1 if random.random() < p else 0
    return result


def availability(distance, car_ratio=0.26, bike_ratio=0.15):
    """
    Returns availability likelihood of transport modes
    Code from Ruo Jia (https://research.chalmers.se/en/person/ruoj)

    @param distance: flaot, trip distance
    @param car_ratio:
    @param bike_ratio:
    @return:
    """
    if distance < 300:
        ratio_walk = 1
        ratio_bike = 0
        ratio_pt = 0
        ratio_car = 0
    else:
        ratio_walk = 1
        ratio_bike = run_monte_carlo(bike_ratio)
        ratio_pt = 1
        ratio_car = run_monte_carlo(car_ratio)

    return ratio_walk, ratio_bike, ratio_pt, ratio_car


def U_walk(walkDistance_x):
    """
    Returns the utility of walking
    Code from Ruo Jia (https://research.chalmers.se/en/person/ruoj)

    @param walkDistance_x:
    @return:
    """
    KmWalkDist = walkDistance_x / 10000
    asc_walk = 2.01
    b_walk_distance = -9.18

    v_walk = asc_walk + b_walk_distance * KmWalkDist

    return v_walk


def U_bike(escooter_distance):
    """
    Returns the utility of biking
    Code from Ruo Jia (https://research.chalmers.se/en/person/ruoj)

    @param escooter_distance:
    @return:
    """

    KmWalkDist = escooter_distance / 10000
    asc_bike = 0.71
    b_walk_distance = -2.19

    v_walk = asc_bike + b_walk_distance * KmWalkDist

    return v_walk


def U_PT(transit_time, waitingTime, mc):
    """
    Returns the utility of taking public transport
    Code from Ruo Jia (https://research.chalmers.se/en/person/ruoj)

    @param transit_time:
    @param waitingTime:
    @param mc:
    @return:
    """
    transitTime = transit_time / 3600
    waitingTime = waitingTime / 3600
    asc_pt = 0
    b_pt_time = -1.38
    b_pt_cost = -0.0388
    b_pt_inner = 0.51
    b_pt_card = 3.18
    v_pt = asc_pt + b_pt_time * (transitTime + waitingTime) + b_pt_cost * 36 + b_pt_inner + b_pt_card * mc

    return v_pt


def U_car(car_time, car_distance, lc, male, md):
    """
    Returns the utility of travelling by car
     Code from Ruo Jia (https://research.chalmers.se/en/person/ruoj)

    @param car_time:
    @param car_distance:
    @param lc:
    @param male:
    @param md:
    @return:
    """
    car_time = car_time / 3600

    asc_car = -0.786
    b_car_time = -0.482
    b_car_cost = -0.0309
    b_car_male = 0.253
    b_lic = 0.575
    b_md = 1.77

    KmCarDist = car_distance / 1000
    fuelprice = 1.6
    CarCost = KmCarDist * fuelprice

    v_car = asc_car + b_car_time * car_time + b_car_cost * CarCost + b_lic * lc + b_car_male * male + b_md * md

    return v_car


def U_taxi(car_time, car_distance, male):
    """
    Returns the utility of travelling by taxi
    Code from Ruo Jia (https://research.chalmers.se/en/person/ruoj)

    @param car_time:
    @param car_distance:
    @param male:
    @return:
    """
    car_time = car_time / 3600

    asc_car = -0.786
    b_car_time = -0.482
    b_car_cost = -0.0309

    KmCarDist = car_distance / 1000
    fuelprice = 17
    CarCost = 61 + KmCarDist * fuelprice

    v_taxi = asc_car + b_car_time * car_time + b_car_cost * CarCost

    return v_taxi
