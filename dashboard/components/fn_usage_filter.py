import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from dash import html, dcc, callback
from dash.dependencies import Input, Output, State

COL_ANT = ['#80b1d3', '#49516f']
COL_POST = ['#f89274', '#BA7872']
COL_SEC = ['#DCEED1']


def usage_frequency_content(df_ids, df_trips, min, max):
    """
    Generates the content for usage frequency analysis
    @param df_ids: DataFrame containing vehicle IDs
    @param df_trips: DataFrame containing trip data
    @param min: Minimum threshold for average daily usage
    @param max: Maximum threshold for average daily usage
    @return: HTML content for the usage frequency section
    """
    scoot_reg = create_scooter_registry(pd.concat(df_ids), pd.concat(df_trips)).to_dict('records')

    cont = (dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Distribution of Providers", className="card-title"),
            dbc.Row(dcc.Graph(figure=range_providers(df_trips)), style={'height': '18vh'}),
            html.Br(),
            html.P(['Total number of vehicles with an average daily usage efficiency between ', str(min), ' and ', str(max), ':'], className='lead'),
            html.H3(str(len(scoot_reg)), className="display-3"),
            html.Br(),
            dcc.Store(id='store-vehicle-ids', data=scoot_reg),
            html.Button("Export Vehicle IDs", id="btn-export-ids"),
            dcc.Download(id="download-export-ids"),

        ], className="card-style"), className='card-container'), width=3),
        dbc.Col([dbc.Card(dbc.CardBody([
            html.H6("Trip Origins of Vehicles", className="card-title"),
            dbc.Row(map_scooter_locations(df_trips), style={'height': '60vh'})], className="card-style"),
            className='card-container')

        ], width=9)]))

    return cont


def get_ids_within_range(data, min, max):
    """
    Filters vehicle IDs based on their average daily usage frequency
    @param data: DataFrame containing trip data
    @param min: Minimum threshold for average daily usage
    @param max: Maximum threshold for average daily usage
    @return: DataFrame with IDs of vehicles within the specified range
    """
    df_usage = data.groupby(['Date', 'id']).speed.count().reset_index().groupby('id').speed.mean().reset_index()
    df_usage = df_usage.rename(columns={'speed': 'daily_freq'})
    return df_usage[(df_usage.daily_freq >= min) & (df_usage.daily_freq <= max)]


def filter_data_via_ids(df_ids, df_trips):
    """
    Filters trip data based on provided vehicle IDs
    @param df_ids: DataFrame containing vehicle IDs
    @param df_trips: DataFrame containing trip data
    @return: Filtered DataFrame containing trips for specified IDs
    """
    filtered_df = pd.merge(df_trips, df_ids, on='id', how='inner')
    return filtered_df


def create_scooter_registry(df_ids, df_trips):
    """
    Creates a registry of scooters with their corresponding types
    @param df_ids: DataFrame containing vehicle IDs
    @param df_trips: DataFrame containing trip data
    @return: DataFrame containing the scooter registry
    """
    registry = pd.merge(df_ids, df_trips[['id', 'type']], on='id', how='left')
    registry = registry.drop_duplicates(subset=['id'])
    return registry


def range_providers(data):
    """
    Generates a pie chart showing the distribution of providers
    @param data: List of DataFrames containing provider data
    @return: Plotly figure object with the distribution of providers
    """
    fig_provider = make_subplots(rows=1, cols=len(data), specs=[[{'type': 'domain'}]] if len(data) == 1 else [
        [{'type': 'domain'}, {'type': 'domain'}]])
    for i, el in enumerate(data):
        prov_distr = el[['type', 'id']].groupby(['type']).id.nunique().reset_index()
        fig_provider.add_trace(
            go.Pie(labels=prov_distr['type'], values=prov_distr['id'], name='before' if i == 0 else 'after',
                   marker=dict(colors=COL_ANT if i == 0 else COL_POST)), 1, i + 1)

    fig_provider.update_traces(hole=.4)
    fig_provider.update_layout(showlegend=False, margin=dict(l=20, r=20, t=10, b=10))

    return fig_provider


def map_scooter_locations(data):
    """
    Generates a map showing scooter locations based on trip origins
    @param data: List of DataFrames containing trip data
    @return: Dash Graph object with the map of scooter locations
    """
    # before, after, providers
    fig = go.Figure()
    for j, d in enumerate(data):
        center = [d['o_lat'].mean(), d['o_lng'].mean()]
        for i, e in enumerate(d.type.unique()):
            prov_temp = d[d.type == e]
            fig.add_trace(go.Scattermapbox(
                lat=prov_temp['o_lat'],
                lon=prov_temp['o_lng'],
                mode='markers',
                name=e,
                text="after" if j == 1 else "",
                marker=dict(
                    color=COL_ANT[i] if j == 0 else COL_POST[i],
                    opacity=0.5
                )
            ))
    fig.update_layout(autosize=True,
                      hovermode='closest',
                      margin=dict(l=20, r=20, t=10, b=20),
                      mapbox=dict(
                          style="carto-positron",
                          center=dict(
                              lat=center[0],
                              lon=center[1]
                          ),
                          zoom=10))

    return dcc.Graph(
        id='bubble-map',
        figure=fig
    )


@callback(
    Output("download-export-ids", "data"),
    Input("btn-export-ids", "n_clicks"),
    State('store-vehicle-ids', 'data'),
    prevent_initial_call=True,
)
def generate_csv_from_ids(n_clicks, store_data):
    """
    Callback function to handle the export of vehicle IDs
    @param n_clicks: Number of clicks on the export button
    @param store_data: Data from the store containing vehicle IDs
    @return: Data for download as a CSV file
    """
    return dcc.send_data_frame(pd.DataFrame(store_data).to_csv, "vehicle_ids.csv")
