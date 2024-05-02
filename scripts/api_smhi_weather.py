import requests
import geopy.distance
import csv
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
import matplotlib.pyplot as plt

api_url = 'https://opendata-download-metobs.smhi.se/api/version/latest/parameter/'


def get_available_stations(params, data_mid_lat, data_mid_lon):
    union_station = []
    for par in params:
        resp = requests.get(api_url + str(par) + '.json').json()
        data_mid = (data_mid_lat, data_mid_lon)

        stations = []
        for x in resp['station']:
            if x['active']:
                station_name = x['name']
                station_mid = (float(x['summary'].split(' ')[1]), float(x['summary'].split(' ')[3]))
                dist = geopy.distance.geodesic(data_mid, station_mid).km
                if dist < 30:  # stations within a radius of 30km of data's midpoint
                    stations.append({'name': station_name, 'dist': dist, 'key': x['key'], 'coord': station_mid})

        stations = sorted(stations, key=lambda d: d['dist'])
        union_station.append(stations)

    # retrieve all close-ish stations which contain all 3 parameters
    avail_stations = [y for y in [x for x in union_station[0] if x in union_station[1]] if y in union_station[2]]
    return avail_stations


def map_stations(stations, data_mid_lat, data_mid_lon):
    fig = go.Figure()

    for s in stations:
        fig.add_scattermapbox(
            lat=[s['coord'][0]],
            lon=[s['coord'][1]],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=14
            ),
            text=[s['key']],
            name=s['name'],
        )

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                      mapbox=dict(style="carto-positron", zoom=8,
                                  center={'lat': data_mid_lat, 'lon': data_mid_lon}))

    fig.show()


def fetch_weather_data(station_key, params, start_date, end_date):
    date_regex = re.compile('^\d{4}-([0][1-9]|1[0-2])-([0][1-9]|[1-2]\d|3[01])$')
    weather_arr = []
    for p in params:
        data = requests.get(api_url + str(p) + '/station/' + str(station_key) + '/period/corrected-archive/data.csv')

        if station_key == 98230 and p == 5:
            # use Stockholm-Observatoriekullen instead of Stockholm-Observatoriekullen A
            data = requests.get(api_url + str(p) + '/station/' + '98210' + '/period/corrected-archive/data.csv')

        content_decode = data.content.decode('utf-8')
        content_csv = list(csv.reader(content_decode.splitlines(), delimiter=';'))
        content_csv = np.array([i for i in content_csv if len(i) == len(content_csv[-1]) and date_regex.match(i[2])])
        df_w = pd.DataFrame()
        df_w['date'] = pd.to_datetime(content_csv[:, 2])
        df_w[str(p)] = content_csv[:, 3]
        weather_arr.append(df_w)

    df_weather = weather_arr[0].merge(weather_arr[1]).merge(weather_arr[2])
    df_weather = df_weather.rename(columns={'19': 'temp_min', '20': 'temp_max', '5': 'precip_mm'})
    for col in ['temp_min', 'temp_max', 'precip_mm']:
        df_weather[col] = pd.to_numeric(df_weather[col], errors='coerce')
    df_weather['temp_avg'] = df_weather[['temp_min', 'temp_max']].mean(axis=1)

    return df_weather[(df_weather.date.dt.date >= start_date) & (df_weather.date.dt.date <= end_date)]


def plot_weather(df_weather, df_missing, city_name):
    fig, ax1 = plt.subplots(figsize=(10, 3), dpi=150)
    ax2 = ax1.twinx()

    ax2.axhline(y=3.5, alpha=0.5, color='blue', ls='--', lw=0.5)
    ax2.bar(df_weather['date'], df_weather['precip_mm'], label='precip_mm', alpha=0.5)

    ax1.axhline(y=8, alpha=0.5, color='grey', ls='-', lw=0.5)
    ax1.bar(df_missing['date'], df_missing['hours'], label='#hours with missing data', alpha=0.5, colour='grey')

    for i in ['temp_min', 'temp_max', 'temp_avg']:
        ax1.plot(df_weather['date'], df_weather[i], label=i)

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Temperature [°C]', color='g')
    ax2.set_ylabel('Daily Precipitation [mm]', color='b')
    ax1.xaxis.set_ticks(df_weather['date'][::2])
    ax1.set_xticklabels(df_weather['date'].dt.date[::2], rotation=75)
    ax1.set_title('Weather in ' + city_name)
    ax1.legend()
    plt.show()