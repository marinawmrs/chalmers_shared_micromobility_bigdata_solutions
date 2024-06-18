import datetime as dt
import numpy as np
import dash_bootstrap_components as dbc
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
from dash import html, dcc
from scipy import stats

from dashboard.components.fn_map_generation import create_aggregated_zones
from dashboard.components.data.var_config import variables

COL_ANT = ['#80b1d3', '#49516f']
COL_POST = ['#f89274', '#BA7872']
COL_SEC = ['#DCEED1', '#6a7862']


def date_reformat(date):
    """
    Reformats ISO-date

    @param date: isodate string (2024-01-01)
    @return: user-friendly string (1 Jan 2024)
    """
    return dt.datetime.strptime(date, '%Y-%m-%d').strftime('%d %b %Y')


def variable_descriptive_stats(drop_val, mode, *series):
    """
    Calculates and displays average of variable over timeframe(s)
    @param drop_val: selected dropdown variable
    @param radio_val: selected radio value of day-mode
    @param arr_daymode: array with data by day-mode
    @param mode: 0 or 1, based on single or compare mode
    @return: graph object
    """
    drop_val_name = variables[drop_val]['name_stat']

    if mode == 0:
        if variables[drop_val]['type'] == 'mean':
            mean_bef = mean_throughout_day(drop_val_name, series[0])

        descr_stats = generate_descriptive_stats(drop_val_name, series[0], series[0])
        var_metric = html.H2("{:.2f}".format(descr_stats['ant_dailyavg']) + " " + variables[drop_val]['metric'],
                             className="display-3") if drop_val_name != 'GHG' else html.H2(
            ["{:.2f}".format(descr_stats['ant_dailyavg']), " ", 'g CO', html.Sub(2), '-eq'],
            className="display-3")

        daily_cumulative_avg = html.Div()
        daily_total_text = html.P('Average Daily ' + drop_val + ' (daily total)', className='lead')
        if drop_val_name == 'reduced_time':
            daily_cumul = series[0].groupby(['isodate'])['reduced_time'].sum().mean() / 3600
            daily_cumulative_avg = html.Div([
                html.Br(),
                daily_total_text,
                html.P(
                    ["{:.2f}".format(daily_cumul), " ", 'h'],
                    className="display-5"),
                html.Br(),
            ])

        if drop_val_name == 'GHG':
            daily_cumul = series[0].groupby(['isodate'])['GHG'].sum().mean() / 1000
            daily_cumulative_avg = html.Div([
                html.Br(),
                daily_total_text,
                html.P(
                    ["{:.2f}".format(daily_cumul), " ", 'kg CO', html.Sub(2), '-eq'],
                    className="display-5"),
                html.Br()
            ])


        if variables[drop_val]['type'] == 'mean':
            fig = graph_mean_throughout_day(drop_val, mean_bef)
        else:
            fig = figure_pdf(drop_val, series[0])
            fig_usage_eff = average_trips_per_scooter(series[0])

        triplevel_addition= ' (trip-level) ' if drop_val_name in ['GHG', 'reduced_time'] else ''
        canvas_content = [html.P('Average Daily ' + drop_val + triplevel_addition, className='lead'),
                          var_metric,
                          daily_cumulative_avg,
                          html.P(['between ' + descr_stats['ant_range']], className='lead'),
                          dcc.Graph(id='graph-hourly-mean', figure=fig, style={'height': '30vh'}),
                          dcc.Graph(id='graph-scooter-trips', figure=fig_usage_eff,
                                    style={'height': '30vh'}) if drop_val == 'Usage Efficiency' else html.Br(),
                          ]

    else:
        descr_stats = generate_descriptive_stats(drop_val_name, series[0], series[1])
        var_metric = variables[drop_val]['metric'] if drop_val_name != 'GHG' else html.Div(['g CO', html.Sub(2), '-eq'])

        if variables[drop_val]['type'] == 'mean':
            mean_bef = mean_throughout_day(drop_val_name, series[0])
            mean_aft = mean_throughout_day(drop_val_name, series[1])
            fig = graph_mean_throughout_day(drop_val, mean_bef, mean_aft)

        else:
            fig = figure_pdf(drop_val, series[0], series[1])
            fig_usage_eff = average_trips_per_scooter(series[0], series[1])

        triplevel_addition = ' (trip-level) ' if drop_val_name in ['GHG', 'reduced_time'] else ''
        decimal_places = "{:.4f}" if "Likelihood" in drop_val else "{:.2f}"
        canvas_content = [
            html.P(drop_val + (' increased by' if descr_stats['change'] > 0 else ' decreased by'), className='lead'),
            html.H2("{:.2f}".format(descr_stats['change']) + "%", className="display-3"),
            html.P([
                'Statistically, this is ' + (
                    'not ' if
                    descr_stats[
                        'temp_ttest'] > 0.001 else '') + 'a significant change. ',
                dbc.Badge("i",
                          id='ttest-info-badge',
                          color='white',
                          text_color='secondary',
                          pill=True,
                          className='border me-1')],
                className='lead'),
            dbc.Tooltip(
                'Based on the dependent T-Test and using a significance level of <0.001, where p= ' + "{:.4f}".format(
                    descr_stats['temp_ttest']) + '.', target='ttest-info-badge'), html.Br(), dbc.Row(
                [dbc.Col([html.P(descr_stats['ant_range']), html.P('Daily Average' + triplevel_addition, className='lead')], width=9),
                 dbc.Col(html.Div(
                     [html.Span(decimal_places.format(descr_stats['ant_dailyavg']) + ' ', style={'color': COL_ANT[0]}),
                      var_metric]), width=3)]), html.Br(), dbc.Row(
                [dbc.Col([html.P(descr_stats['post_range']), html.P('Daily Average' + triplevel_addition, className='lead')], width=9),
                 dbc.Col(html.Div(
                     [html.Span(decimal_places.format(descr_stats['post_dailyavg']) + ' ', style={'color': COL_POST[0]}),
                      var_metric]), width=3)]), html.Br(),
            dcc.Graph(id='graph-hourly-mean', figure=fig, style={'height': '30vh'}),
            dcc.Graph(id='graph-scooter-trips', figure=fig_usage_eff,
                      style={'height': '30vh'}) if drop_val == 'Usage Efficiency' else html.Br(),
        ]
    return dbc.Col(dbc.Card(dbc.CardBody(canvas_content, className="card-style"), className='card-container'))


def mean_throughout_day(sel_var, data):
    """
    Calculates the mean and standard deviation for every hour of the day

    @param sel_var: variable to calculate stats for
    @param data: df
    @return: dataframe with hourly mean and st
    """
    mean_bef = data.groupby('hour')[sel_var].mean().reset_index()
    mean_bef['std'] = data.groupby('hour')[sel_var].std()
    mean_bef['lower'] = mean_bef[sel_var] - 0.5 * mean_bef['std']
    mean_bef['upper'] = mean_bef[sel_var] + 0.5 * mean_bef['std']

    return mean_bef


def graph_mean_throughout_day(drop_val, *series):
    """
    Creates Graph showing the mean and standard deviation per hour of the day for one variable
    @param drop_val: selected variable
    @param series: tuple of input df (i.e. mean_before, mean_after)
    @return: figure with mean & std
    """
    fig = go.Figure([])
    counter = 0
    for s in series:
        h = COL_ANT[0].lstrip('#') if counter == 0 else COL_POST[0].lstrip('#')
        fig.add_trace(go.Scatter(x=s.hour, y=s[variables[drop_val]['name_stat']] + 0.5 * s.lower, mode='lines',
                                 line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=s.hour, y=s[variables[drop_val]['name_stat']] - 0.5 * s.lower, mode='lines',
                                 line=dict(width=0), showlegend=False, hoverinfo='skip',
                                 fillcolor='rgba' + str(tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)))[:-1] + ', 0.2)',
                                 fill='tonexty', ))
        fig.add_trace(go.Scatter(x=s.hour, y=s[variables[drop_val]['name_stat']],
                                 line=dict(color='rgb' + str(tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)))),
                                 mode='lines', showlegend=False,
                                 name=drop_val + ' (before)' if counter == 0 else ' (after)'))
        counter += 1

    fig.update_layout(title="Average " + drop_val, xaxis_title="hour of the day",
                      yaxis_title=variables[drop_val]['metric'])
    return fig


def figure_pdf(drop_val, *series):
    """
    Creates Graph showing the PDF
    @param drop_val: selected variable
    @param series: tuple of input df (i.e. mean_before, mean_after)
    @return: figure with PDF of variable
    """
    fig = go.Figure()

    if variables[drop_val]['name_stat'] == 'demand':
        c = 0
        for s in series:
            fig.add_trace(go.Histogram(x=s.o_time.apply(lambda x: x.hour), name='before' if c == 0 else 'after',
                                       xbins=dict(start=0, end=24, size=1), histnorm='probability',
                                       marker_color=COL_ANT[0] if c == 0 else COL_POST[0]))
            c += 1
        fig.update_layout(title_text='Trips per hour', xaxis_title='hour of the day',
                          yaxis_title='Probability', )

    if variables[drop_val]['name_stat'] == 'vehicle':
        c = 0
        for s in series:
            fig.add_trace(go.Bar(x=s.groupby(['isodate']).id.nunique().reset_index().isodate,
                                 y=s.groupby(['isodate']).id.nunique().reset_index().id,
                                 name='before' if c == 0 else 'after',
                                 marker_color=COL_ANT[0] if c == 0 else COL_POST[0]))
            c += 1
        fig.update_layout(title_text='Vehicles per Day', xaxis_title='date', yaxis_title='Number of vehicles', )

    if variables[drop_val]['name_stat'] == 'freq':
        c = 0
        for s in series:
            fig.add_trace(go.Histogram(x=s.groupby(['id']).o_time.count(), name='before' if c == 0 else 'after',
                                       histnorm='probability', marker_color=COL_ANT[0] if c == 0 else COL_POST[0]))
            c += 1
        fig.update_layout(title_text='Distribution of Usage Efficiency', xaxis_title=variables[drop_val]['metric'],
                          yaxis_title='Probability', )

    return fig


def average_trips_per_scooter(*series):
    """
    Creates Graph showing the average number of trips per scooter (usage efficiency) & std per day
    @param series: tuple of input df (i.e. mean_before, mean_after)
    @return: figure
    """
    fig = go.Figure()

    counter = 0
    for s in series:
        h = COL_ANT[0].lstrip('#') if counter == 0 else COL_POST[0].lstrip('#')
        arr_mean = s.groupby(['isodate', 'id']).o_time.count().reset_index().groupby(['isodate'])[
            'o_time'].mean().reset_index()
        arr_std = s.groupby(['isodate', 'id']).o_time.count().reset_index().groupby(['isodate'])[
            'o_time'].std().reset_index()
        fig.add_trace(
            go.Scatter(x=arr_mean.isodate, y=arr_mean.o_time + 0.5 * arr_std.o_time, mode='lines', line=dict(width=0),
                       showlegend=False, hoverinfo='skip'))
        fig.add_trace(
            go.Scatter(x=arr_mean.isodate, y=arr_mean.o_time - 0.5 * arr_std.o_time, mode='lines', line=dict(width=0),
                       showlegend=False, hoverinfo='skip',
                       fillcolor='rgba' + str(tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)))[:-1] + ', 0.2)',
                       fill='tonexty', ))
        fig.add_trace(go.Scatter(x=arr_mean.isodate, y=arr_mean.o_time,
                                 line=dict(color='rgb' + str(tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)))),
                                 mode='lines', showlegend=False,
                                 name='trips per vehicle' + ' (before)' if counter == 0 else ' (after)'))
        counter += 1

    fig.update_layout(title_text='Average Usage Efficiency per Day', xaxis_title='date',
                      yaxis_title='Trips per Vehicle')

    return fig


def generate_descriptive_stats(sel_var, ant, post):
    """
    Creates descriptive statistics (daily average, timeframes, t-test)

    @param sel_var: variable to calculate stats for
    @param ant: tuple of input df (i.e. mean_before)
    @param post: tuple of input df (i.e. mean_after)
    @return: df with descriptive stats for ant and post df
    """
    stat_dict = dict()
    stat_dict['ant_range'] = date_reformat(ant['isodate'].min()) + ' - ' + date_reformat(ant['isodate'].max())
    stat_dict['post_range'] = date_reformat(post['isodate'].min()) + ' - ' + date_reformat(post['isodate'].max())

    if sel_var == 'demand':
        stat_dict['ant_dailyavg'] = ant.groupby(['isodate']).size().mean()
        stat_dict['post_dailyavg'] = post.groupby(['isodate']).size().mean()
        stat_dict['temp_ttest'] = stats.ttest_ind(ant.groupby(['isodate']).id.size().reset_index().id,
                                                  post.groupby(['isodate']).id.size().reset_index().id,
                                                  equal_var=True).pvalue

    elif sel_var == 'vehicle':
        stat_dict['ant_dailyavg'] = ant.groupby(['isodate']).id.nunique().mean()
        stat_dict['post_dailyavg'] = post.groupby(['isodate']).id.nunique().mean()
        stat_dict['temp_ttest'] = stats.ttest_ind(ant.groupby(['isodate']).id.nunique().reset_index().id,
                                                  post.groupby(['isodate']).id.nunique().reset_index().id,
                                                  equal_var=True).pvalue

    elif sel_var == 'freq':
        stat_dict['ant_dailyavg'] = ant.groupby(['isodate', 'id']).size().reset_index()[0].mean()
        stat_dict['post_dailyavg'] = post.groupby(['isodate', 'id']).size().reset_index()[0].mean()
        stat_dict['temp_ttest'] = stats.ttest_ind(ant.groupby(['isodate', 'id']).size().reset_index()[0],
                                                  post.groupby(['isodate', 'id']).size().reset_index()[0],
                                                  equal_var=True).pvalue

    else:
        # get daily averages & ttest
        stat_dict['ant_dailyavg'] = ant.groupby(['isodate'])[sel_var].mean().mean()
        stat_dict['post_dailyavg'] = post.groupby(['isodate'])[sel_var].mean().mean()
        stat_dict['temp_ttest'] = stats.ttest_ind(ant[sel_var], post[sel_var], equal_var=True).pvalue

    # calculate change
    stat_dict['change'] = (stat_dict['post_dailyavg'] - stat_dict['ant_dailyavg']) / stat_dict['ant_dailyavg'] * 100

    return stat_dict


def map_generation_pipeline(drop_val, radio_aggr, grid_choice, *series):
    """
    Run consecutive actions for generating the map
    @param radio_aggr: aggregation choice
    @param drop_val: selected variable
    @param series: dataframe, (before/after)
    @return:
    """
    cols = ['o_lng', 'o_lat', 'id', 'Date']
    if drop_val not in ['demand', 'vehicle', 'freq']:
        cols.append(drop_val) if isinstance(drop_val, str) else cols.extend(drop_val)

    zone_arr = []
    for s in series:
        gdf = gpd.GeoDataFrame(s[cols], geometry=gpd.points_from_xy(s['o_lng'], s['o_lat']))
        gdf = gdf.set_crs(4326)

        zone_aggr = create_aggregated_zones(gdf, drop_val, radio_aggr, grid_choice)
        zone_aggr = zone_aggr[zone_aggr['demand_total'] > 0]
        zone_aggr['id'] = zone_aggr.index
        zone_arr.append(zone_aggr)

    return zone_arr


def variable_map_card(drop_var, radio_var, arr_daymode, mode, aggrchoice):
    """
    Creates card including the map with zones
    @param drop_var: selected dropdown variable
    @param radio_var:  selected radio value of day-mode
    @param arr_daymode: array with data by day-mode
    @param mode: 0 or 1, based on single or compare mode
    @param aggrchoice: aggregation fucntion for map zones
    @return: div content
    """
    drop_val_name = variables[drop_var]['name_stat']

    print(aggrchoice)
    src_map = map_generation_pipeline(drop_val_name, aggrchoice,
                                      arr_daymode[radio_var - 1]) if mode == 0 else map_generation_pipeline(
        drop_val_name, aggrchoice, arr_daymode[radio_var - 1][0], arr_daymode[radio_var - 1][1])
    div_map = dbc.Card([dbc.CardBody([
        html.Iframe(id='spatial_map', srcDoc=src_map, style={'width': '100%', 'height': '100%', 'overflow': 'hidden'})],
        className="card-style", style={'height': '100%'})], className='card-container', style={'height': '70vh'})

    return div_map


def violin_plot_level(level, drop_val, *series):
    """
    Creates violin plot based on aggregated zones
    @param level: string, trip or zone level
    @param drop_val: selected variable
    @param series: (geo-)dataframe, (before/after)
    @return: graph object
    """
    drop_val_name = variables[drop_val]['name_stat']
    fig = go.Figure()

    for i, s in enumerate(series):
        if level == 'Zone':
            s = pd.DataFrame(s[drop_val_name]).dropna()

        fig.add_trace(go.Violin(y=s[drop_val_name], box_visible=False, points=False,
                                meanline_visible=True, opacity=0.6, fillcolor=COL_ANT[0] if i == 0 else COL_POST[0],
                                x0='Timeframe 1' if i == 0 else 'Timeframe 2',
                                meanline_color=COL_ANT[1] if i == 0 else COL_POST[1]))
        violin_annotation(fig, i, s, drop_val_name)

    plot_title = '{}-level'.format(
        level) if level == 'Trip' else '{}-level <br><sup>(selected zone aggregator function)</sup>'.format(level)
    fig.update_layout(title=plot_title, yaxis_title=variables[drop_val]['metric'])
    fig.update_traces(line_width=0, meanline_width=2, points=False, selector=dict(type='violin'))

    return fig


def violin_annotation(fig, i, s, variable):
    """
    Appends annotations to violin plot (positive impact % and mean)
    @param fig: graph object
    @param i: counter from series s
    @param s: series s
    @param variable: selected variable
    @return: graph object
    """
    fig.add_annotation(x=i, yref='paper', y=1,
                       text="Positive impact: {:.2%}".format(
                           (s[variable].dropna() > 0).sum() / len(s[variable].dropna())),
                       showarrow=False,
                       font=dict(
                           color=COL_SEC[1]
                       ))

    fig.add_annotation(x=i, yref='paper', y=0,
                       text="Mean: {:.2f}".format(s[variable].mean()) + "<br>Median: {:.2f}".format(s[variable].median()),
                       showarrow=False,
                       font=dict(
                           color=COL_ANT[1] if i == 0 else COL_POST[1]
                       ))
    return fig


def figure_modal_subs(level, *series):
    """
    Creates bar chart of modal substitution rate
    @param level: string, level or zone
    @param series: single or both timeframes, either timeseries or aggregated zones
    @return: graph object
    """
    subs_mode = pd.DataFrame()
    arr = ['Modal substitution' if len(series) == 1 else 'Timeframe 1', 'Timeframe 2']
    modes = ['P_walk', 'P_car', 'P_taxi', 'P_bike', 'P_PT']

    for i, s in enumerate(series):
        subs_mode[arr[i]] = s[modes].dropna().idxmax(axis=1).reset_index().groupby([0]).count() / \
                            s[modes].dropna().shape[0]
    subs_mode = subs_mode.transpose().reset_index(names='interval').fillna(0)

    fig = go.Figure()

    for i in modes:
        if i in subs_mode.columns:
            fig.add_trace(
                go.Bar(name=i, x=subs_mode.interval, y=subs_mode[i], hovertemplate='%{y:.2%}', text=subs_mode[i], ))
    fig.update_layout(showlegend=True, barmode='stack', title='{}-level'.format(level),
                      yaxis_title='Substitution Likelihood')
    fig.update_traces(texttemplate='%{text:.2%}', textposition='auto')
    return fig


def figure_balance_point(*series):
    """
    Creates figure for balance point of vehicle GHG emissions
    @param series: df
    @return: figure
    """

    fig = go.Figure()

    for j, s in enumerate(series):
        if len(series) == 1:
            s['distance_round'] = np.floor(s.escooter_distance / 1000)
            for x in range(0, 10):
                sx = s[s.distance_round == x]
                mean_l = []
                for i in np.linspace(0, 100, 11):
                    sx['GHG2'] = sx.car_distance / 1000 * 160.7 * (
                                sx.P_car + sx.P_taxi) + sx.transit_transitdistance / 1000 * 16.04 * sx.P_PT + sx.escooter_distance / 1000 * 37.0 * sx.P_bike - sx.escooter_distance / 1000 * i
                    mean_l.append(sx.GHG2.mean())

                fig.add_trace(go.Scatter(name=str(x) + '-' + str(x + 1), x=np.linspace(0, 100, 11), y=np.array(mean_l),
                                         mode='lines', hoverinfo='none', opacity=0.3, line={'dash': 'dash'}))

        mean_l = []
        for i in np.linspace(0, 100, 11):
            s['GHG2'] = s.car_distance / 1000 * 160.7 * (
                        s.P_car + s.P_taxi) + s.transit_transitdistance / 1000 * 16.04 * s.P_PT + s.escooter_distance / 1000 * 37.0 * s.P_bike - s.escooter_distance / 1000 * i
            mean_l.append(s.GHG2.mean())
        zero_point = round(get_intercept(0, mean_l[0], 100, mean_l[-1]), 2)

        fig.add_trace(
            go.Scatter(name='Average', x=np.linspace(0, 100, 11), y=np.array(mean_l), mode='lines', hoverinfo='none',
                       line={'color': COL_ANT[0] if j == 0 else COL_POST[0]}))
        fig.add_trace(go.Scatter(
            x=[zero_point],
            y=[0],
            mode="markers",
            name="Balance Point",
            text=[str(zero_point)] if j == 0 else ['Timeframe 2'],
            textposition="bottom center",
            line={'color': COL_ANT[0] if j == 0 else COL_POST[0]},
            showlegend=False,
            hovertemplate='%{x:.2f}<extra>Balance Point</extra>',
        ))
    fig.update_layout(showlegend=True if len(series) == 1 else False,
                      xaxis_title='Emission factor of SES (CO<sub>2</sub> g/km)',
                      yaxis_title='GHG Reduction (CO<sub>2</sub>/g)', legend_title_text='Distance (km)')

    return fig


def get_intercept(x1,y1,x2,y2):
    """
    Calculates 0-intercerpt
    @param x1: coord
    @param y1: coord
    @param x2: coord
    @param y2: coord
    @return: coord, x where 0-intercerpt
    """
    k = (y1-y2)/(x1-x2)
    b = y1 - k*x1
    return -b/k

