import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import html, dcc
from plotly.subplots import make_subplots

from dashboard.components.fn_descriptive_statistics import date_reformat

COL_ANT = ['#80b1d3', '#49516f']
COL_POST = ['#f89274', '#BA7872']
COL_SEC = ['#DCEED1']


def overview_compute_metadata(date_start, date_end, data, mode):
    """
    returns card for high-level metadata
    @param date_start: timeframe start
    @param date_end: timeframe end
    @param data: data
    @param mode: 0 or 1, based on compare mode
    @return:
    """
    return dbc.Col(dbc.Card(dbc.CardBody([html.H6("Data Summary", className="card-title"),
                                          html.P(date_reformat(date_start) + ' to ' + date_reformat(date_end),
                                                 style={'color': COL_ANT[0] if mode == 0 else COL_POST[0]},
                                                 className='lead'), html.Br(),
                                          html.P('Recorded Trips: ' + '{:,}'.format(data.id.count()),
                                                 className="card-text"),
                                          html.P('Vehicles: ' + '{:,}'.format(data.id.nunique()),
                                                 className="card-text"),
                                          html.P('Number of days: ' + '{:,}'.format(data.isodate.nunique()),
                                                 className="card-text"),

                                          ], className="card-style", style={'min-height': '25vh'}),
                            className='card-container'), width=3)


def overview_trips_per_day(data_before, data_after, mode):
    """
    Calculates trips per day and generates histogram
    @param data_before: single (or first) datarange
    @param data_after: if applicable, second datarange for comaprison
    @param mode: 0 or 1, based on single or compare mode
    @return: graph object
    """

    fig_datapoints = go.Figure()
    fig_datapoints.add_trace(go.Histogram(x=data_before['isodate'], name='before', marker_color=COL_ANT[0]))
    if mode == 1 and data_after is not None:
        fig_datapoints.add_trace(go.Histogram(x=data_after['isodate'], name='after', marker_color=COL_POST[0]))

    return dbc.Col(dbc.Card(dbc.CardBody([html.H6("Number of Trips per Day", className="card-title"), dcc.Graph(
        figure=fig_datapoints.update_layout(bargap=0.2),
        style={'height': '45vh'})], className="card-style"), className='card-container'), width=7)


def overview_trips_by_daytype(data_before, data_after, mode):
    """
    Calculates histogram of trips by different weekdays  and generates histogram
    @param data_before: single (or first) datarange
    @param data_after: if applicable, second datarange for comaprison
    @param mode: 0 or 1, based on single or compare mode
    @return: graph object
    """
    fig_daymode = go.Figure()
    fig_daymode.add_trace(go.Histogram(x=data_before['day'], name='before', marker_color=COL_ANT[0]))
    if mode == 1 and data_after is not None:
        fig_daymode.add_trace(go.Histogram(x=data_after['day'], name='after', marker_color=COL_POST[0]))
    fig_daymode.update_xaxes(categoryorder='array',
                             categoryarray=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
                                            'Sunday'])

    return dbc.Col(dbc.Card(dbc.CardBody([html.H6("Number of Trips by Weekday", className="card-title"),
                                          dcc.Graph(figure=fig_daymode,
                                                    style={'height': '45vh'})], className="card-style"),
                            className='card-container'), width=5)


def overview_providers(data_before, data_after, mode):
    """
    Calculates and displays pie chart of providers
    @param data_before: single (or first) datarange
    @param data_after: if applicable, second datarange for comaprison
    @param mode: 0 or 1, based on single or compare mode
    @return: graph object
    """
    prov_distr = data_before[['type', 'id']].groupby(['type']).id.count().reset_index()
    fig_provider = make_subplots(rows=1, cols=1 + mode, specs=[[{'type': 'domain'}]] if mode == 0 else [
        [{'type': 'domain'}, {'type': 'domain'}]])
    fig_provider.add_trace(
        go.Pie(labels=prov_distr['type'], values=prov_distr['id'], name='before', marker=dict(colors=COL_ANT)), 1,
        1)
    if data_after is not None and mode == 1:
        prov_distr = data_after[['type', 'id']].groupby(['type']).id.count().reset_index()
        fig_provider.add_trace(
            go.Pie(labels=prov_distr['type'], values=prov_distr['id'], name='after', marker=dict(colors=COL_POST)),
            1, 2)
    fig_provider.update_traces(hole=.4)
    fig_provider.update_layout(margin=dict(l=20, r=20, t=10, b=20))

    return dbc.Col(dbc.Card(dbc.CardBody([html.H6("Distribution of Operators", className="card-title"),
                                          dcc.Graph(figure=fig_provider.update_layout(showlegend=False),
                                                    style={'height': '18vh'})],
                                         className="card-style", style={'min-height': '25vh'}),
                            className='card-container'),
                   width=(9 if mode == 0 else 6))
