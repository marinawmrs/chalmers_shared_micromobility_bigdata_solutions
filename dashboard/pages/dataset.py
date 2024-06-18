import datetime as dt

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import dashboard.components.additional_styles as stylz
from dashboard.components.fn_dataset_overview import overview_compute_metadata, overview_providers, overview_trips_per_day, \
    overview_trips_by_daytype
from dashboard.components.fn_descriptive_statistics import map_generation_pipeline, violin_plot_level, \
    figure_modal_subs, figure_balance_point, variable_descriptive_stats
from dashboard.components.fn_map_generation import generate_single_map, generate_compare_map, generate_modal_subs_map
from dashboard.components.fn_preprocessing import query_data_within_range, get_day_list
from dashboard.components.fn_usage_filter import get_ids_within_range, filter_data_via_ids, usage_frequency_content
from dashboard.components.data.var_config import providers
from dashboard.components.data.var_config import variables

dash.register_page(__name__, path='/')

COL_ANT = ['#80b1d3', '#49516f']
COL_POST = ['#f89274', '#BA7872']
COL_SEC = ['#DCEED1']

radio_compare_mode = html.Div([dcc.RadioItems(id="radios-compare-mode",
                                              options=[{"label": "System performance analysis", "value": 1},
                                                       {"label": "Comparison analysis", "value": 2}, ], value=1,
                                              inputStyle={"margin-right": "10px"},
                                              labelStyle={'margin': '10px'}, persistence=True,
                                              persistence_type='session')])

daterange_interval_1 = html.Div([
    dcc.DatePickerRange(id='datepicker-interval-1', display_format='DD.MM.YYYY', persistence=True,
                        persistence_type='session', min_date_allowed=dt.date(2022, 8, 5),
                        # mindate+1 from uplaoded data
                        max_date_allowed=dt.date(2022, 10, 5), # update according to data
                        initial_visible_month=dt.date(2022, 8, 5)
                        )])
daterange_interval_2 = html.Div([html.P("Select second timeframe for comparison", className='lead'),
                                 dcc.DatePickerRange(id='datepicker-interval-2', display_format='DD.MM.YYYY',
                                                     persistence=True,
                                                     persistence_type='session', min_date_allowed=dt.date(2022, 8, 5),
                                                     max_date_allowed=dt.date(2022, 10, 5),
                                                     initial_visible_month=dt.date(2022, 9, 5)
                                                     ), ], id='container-datepicker-interval-2',
                                style={'display': 'none'})

provider_dropdown = html.Div([
    dcc.Dropdown(options=list(providers.keys()), id='dropdown-providers', value=list(providers.keys()), clearable=False,
                 multi=True, persistence=True, persistence_type='session', ), ])

sidebar = html.Div(
    [html.H2('Data Analysis', id='sidebar-title'),
     html.Br(), html.P("Select function", className='lead'), radio_compare_mode,
     html.Br(), html.P("Select timeframe", className='lead'),
     daterange_interval_1, daterange_interval_2,
     html.Br(), html.P("Select operators", className='lead'),
     provider_dropdown, html.Button('Load Data', id='button-load-data', style={'display': 'inline-block',
                                                                               'verticalAlign': 'middle',
                                                                               "minWidth": "200px",
                                                                               "marginTop": "10px"}),
     dcc.Store(id='store-filter-params', storage_type='session'), ], style=stylz.SIDEBAR_STYLE, )

opts = []
for i, e in enumerate(list(variables.keys())):
    opts.append(e)

variable_nav = dbc.Col(dbc.Card(dbc.CardBody([dbc.Row([  # Dropdown: Variable Selection
    dbc.Col(html.Div([html.P("Inspect Variable", className="lead"),
                      dcc.Dropdown(options=opts, id='dropdown-var-nav', value=None, clearable=True), ]), width=4),
    # Radio: Weekday Selection
    dbc.Col(html.Div([html.P("Select Weekday Subset", className="lead"),
                      dbc.RadioItems(id="radio-var-daymode", className="btn-group", inputClassName="btn-check",
                                     labelClassName="btn btn-outline-secondary btn-lg", labelCheckedClassName="active",
                                     options=[{"label": "General", "value": 1}, {"label": "Mon-Thur", "value": 2},
                                              {"label": "Fri", "value": 3}, {"label": "Sat-Sun", "value": 4}, ],
                                     value=1,
                                     style={'width': '100%'}, labelStyle={'width': '100%'}, ), ],
                     className="radio-group",
                     style={"width": "100%"}, id='container-radio-var-daymode'), width=4),
    dbc.Col([  # Dropdown for map OR slider for usage frequency
        html.Div([html.P("Select Map Aggregator", className="lead"),
                  dcc.Dropdown(id="radios-aggr-choice", clearable=False),
                  dcc.RadioItems(id="radios-map-grid-choice",
                                 options=[{"label": "Square Zones", "value": 'sqr'},
                                          {"label": "Hexagonal Zones", "value": 'hexa'}, ], value='sqr',
                                 inputStyle={"margin-right": "10px"}, labelStyle={'margin': '10px'}, persistence=True,
                                 persistence_type='session')],
                 id='container-radios-aggr-choice'),
        html.Div([html.P("Show all vehicles where Usage Efficiency is in the range of:", className="lead"),
                  dcc.RangeSlider(0, 10, value=[0, 1.5], marks=None,
                                  tooltip={"placement": "bottom", "always_visible": True, },
                                  id='rangeslider-inspect-freq'), ], id='nav-inspect-scooter')], width=4), ])],
    className="card-style"), className='card-container', id='nav-inspect-variable'), width=12)

layout = html.Div([
    variable_nav,
    dcc.Loading(id='loading', children=[
        html.Div(id='container-overview-subcontent'),
        html.Div([dcc.Tabs([
            dcc.Tab(html.Div(id='container-variable-descr-map'), label='Filter Scooters by Usage Efficiency',
                    value='filter-descr-map', style=stylz.tab_style, selected_style=stylz.tab_selected_style),
            dcc.Tab(html.Div(id='container-variable-usagefreq'), label='Filter Scooters by Usage Efficiency',
                    value='filter-usage-eff', style=stylz.tab_style, selected_style=stylz.tab_selected_style)
        ], value='filter-descr-map', id='tabs-usage-eff', style=stylz.tabs_styles)
        ], id='container-variable-subcontent')])], style=stylz.CONTENT_STYLE_OVERVIEW)


@callback(Output('container-datepicker-interval-2', 'style'), Input("radios-compare-mode", "value"), )
def show_daterange_picker_2(value):
    """
    Renders second datepicker range for compare mode
    @param value: single/compare mode
    @return: explanation sentence & date picker for compare interval
    """
    if value == 1:
        return {'display': 'none'}
    if value == 2:
        return {'display': 'block'}


@callback(Output('container-radio-var-daymode', 'style'), Input("dropdown-var-nav", "value"), )
def show_radio_daymode(value):
    """
    Shows/hides radio for selecting daymode only if variable is selected
    @param value: dropdown - variable selection
    @return: display style of radio
    """
    if value is None:
        return {'display': 'none'}
    else:
        return {'display': 'block'}


@callback(Output('nav-inspect-variable', 'style'), Input("store-filter-params", "data"), )
def show_nav_inspect_variable(value):
    """
    Shows/hides navigation bar for inspecting a variable
    @param value: dropdown - variable selection
    @return: display style of radio
    """
    if value is None:
        return {'display': 'none'}
    else:
        return {'display': 'block'}


@callback(Output('container-radios-aggr-choice', 'style'), Output('nav-inspect-scooter', 'style'),
          Input("dropdown-var-nav", "value"), Input('tabs-usage-eff', 'value'))
def show_radio_aggrchoice(value, tab_val):
    """
    Shows/hides radio for selecting aggregation function only if variable is selected
    @param value: selected var
    @param tab_val: currently active tab
    @return: aggregation dropdown/rangeslider visibility
    """
    if value is None:
        return {'display': 'none'}, {'display': 'none'}
    else:
        if value == 'Usage Efficiency':
            if tab_val == 'filter-usage-eff':
                return {'display': 'none'}, {'display': 'block'}
        return {'display': 'block'}, {'display': 'none'}


@callback(Output('tabs-usage-eff', 'style'), Input("dropdown-var-nav", "value"), )
def show_tabs_userfreq(value):
    """
    Change visibility for tabs in usage efficiency
    @param value: selected variable
    @return: visibility setting
    """
    if value == 'Usage Efficiency':
        return stylz.tabs_styles
    else:
        return {'display': 'none'}


@callback(Output('radios-aggr-choice', 'options'), Output('radios-aggr-choice', 'value'),
          Input('dropdown-var-nav', 'value'))
def update_radio_aggrchoice_options(drop_var):
    """
    Updates available options for map aggregation functions
    @param drop_var: selected var
    @return: list of choices
    """
    if not drop_var:
        raise PreventUpdate

    temp = variables[drop_var]
    opts = []

    # create radio with all options for the map
    for i in range(len(temp['expl'])):
        opts.append({'label': temp['expl'][i], 'value': temp['name_map'][i]})

    return opts, temp['name_map'][0]


@callback(Output('button-load-data', 'disabled'),
          [Input("datepicker-interval-1", "start_date"), Input("datepicker-interval-1", "end_date"),
           Input("dropdown-providers", "value"), Input("radios-compare-mode", "value"),
           Input("datepicker-interval-2", "start_date"), Input("datepicker-interval-2", "end_date"), ])
def disable_dataload_button(start1, end1, prov, mode, start2, end2):
    """
    Disables "Load Data" Button if not sufficient inputs

    @param start1: daterange1
    @param end1: daterange1
    @param prov: provider list
    @param mode: single/compare
    @param start2: daterange2
    @param end2: daterange2
    @return: (dis)abled date of button
    """
    if start1 is not None and end1 is not None and prov is not None:
        if mode == 1:
            return False
        elif start2 is not None and end2 is not None:
            return False

    return True


@callback(Output('store-filter-params', 'data'), Input("button-load-data", "n_clicks"),
          [State("datepicker-interval-1", "start_date"), State("datepicker-interval-1", "end_date"),
           State("dropdown-providers", "value"), State("radios-compare-mode", "value"),
           State("datepicker-interval-2", "start_date"), State("datepicker-interval-2", "end_date")])
def store_current_filters(value, start1, end1, prov, mode, start2, end2):
    """
    Store filter parameters so they can be used across the dashboard

    @param value: button click
    @param start1: daterange1
    @param end1: daterange1
    @param prov: provider list
    @param mode: single/compare
    @param start2: daterange2
    @param end2: daterange2
    @return: (dis)abled date of button
    """
    if value is None:
        raise PreventUpdate

    data = dict()
    data['mode'] = mode
    data['start_date_before'] = start1
    data['end_date_before'] = end1
    data['providers'] = prov
    if mode == 2:
        data['start_date_after'] = start2
        data['end_date_after'] = end2

    return data


@callback(Output('tabs-usage-eff', 'value'), Input('dropdown-var-nav', 'value'))
def reset_tabs_to_descr(drop_var):
    """
    Resets tabs to descriptive view when changing variable
    @param drop_var: chosen variable
    @return: default value for tabs
    """
    if drop_var is not None and drop_var != ['Usage Efficiency']:
        return 'filter-descr-map'


@callback(Output('container-overview-subcontent', 'children', allow_duplicate=True),
          Output('container-overview-subcontent', 'style'),
          Output('container-variable-subcontent', 'style'),
          Output('container-variable-descr-map', 'children'),
          Output('container-variable-usagefreq', 'children'),
          [Input("store-filter-params", "modified_timestamp"),
           Input('dropdown-var-nav', 'value'),
           Input('radio-var-daymode', 'value'),
           Input('radios-aggr-choice', 'value'),
           Input('tabs-usage-eff', 'value'),
           Input('rangeslider-inspect-freq', 'value'),
           Input('radios-map-grid-choice', 'value')],
          [State("store-filter-params", "data")],
          prevent_initial_call=True)
def query_data(ts, drop_var, radio_var, aggrchoice, tabs_val, scooter_slide, grid_choice, data):
    """
    Query data from database given selected parameters, run functions, render plots

    @param ts: timestamp
    @param drop_var: selected var
    @param radio_var: selected daymode
    @param aggrchoice: selected mode for map aggregation
    @param tabs_val: currently selected tab (descr vs usage efficiency)
    @param scooter_slide: rangeslider values, for usage efficiency
    @param data: stored high level parameters
    @return: page content
    """

    data = data
    arr_providers = [providers[i] for i in data.get('providers')]
    day_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    data_after = None

    if drop_var is None:
        data_before = query_data_within_range(arr_providers, data.get('start_date_before'), data.get('end_date_before'),
                                              ['id', 'isodate', 'type', 'day'], day_list)
        if data.get('mode') == 2:
            data_after = query_data_within_range(arr_providers, data.get('start_date_after'),
                                                 data.get('end_date_after'), ['id', 'isodate', 'type', 'day'], day_list)

        card_meta_before = overview_compute_metadata(data.get('start_date_before'), data.get('end_date_before'),
                                                     data_before, 0)
        card_meta_after = None if data.get('mode') == 1 else overview_compute_metadata(data.get('start_date_after'),
                                                                                       data.get('end_date_after'),
                                                                                       data_after, 1)
        pie_providers = overview_providers(data_before, data_after, data.get('mode') - 1)
        graph_trips_per_day = overview_trips_per_day(data_before, data_after, data.get('mode') - 1)
        graph_trips_day_type = overview_trips_by_daytype(data_before, data_after, data.get('mode') - 1)

        first_row = dbc.Row([card_meta_before, pie_providers]) if data.get('mode') == 1 else dbc.Row(
            [card_meta_before, card_meta_after, pie_providers])

        return [first_row, dbc.Row([graph_trips_per_day, graph_trips_day_type])], {'display': 'block'}, {
            'display': 'none'}, [], []

    if drop_var is not None:

        # specify required columns
        drop_val_name = variables[drop_var]['name_stat']
        column_names = ['id', 'o_time', 'o_lng', 'o_lat', 'isodate', 'Date', 'day', 'hour', 'type', 'speed',
                        'car_distance', 'transit_transitdistance', 'escooter_distance', 'P_car', 'P_taxi', 'P_PT', 'P_bike']
        if drop_val_name not in ['demand', 'vehicle', 'freq']:
            column_names.append(drop_val_name) if isinstance(drop_val_name, str) else column_names.extend(drop_val_name)

        column_names = list(dict.fromkeys(column_names))
        day_list = get_day_list(radio_var)

        data_before = query_data_within_range(arr_providers, data.get('start_date_before'), data.get('end_date_before'),
                                              column_names, day_list)
        if data.get('mode') == 2:
            data_after = query_data_within_range(arr_providers, data.get('start_date_after'),
                                                 data.get('end_date_after'), column_names, day_list)

        if aggrchoice is None:
            aggrchoice = variables[drop_var]['name_map'][0]

        aggr_zone = map_generation_pipeline(drop_val_name, aggrchoice, grid_choice, data_before) if data.get(
            'mode') - 1 == 0 else map_generation_pipeline(drop_val_name, aggrchoice, grid_choice, data_before,
                                                          data_after)

        if drop_var == 'Modal Substitution Rate':
            aggr_zone[0]['modal_sub'] = aggr_zone[0][['P_walk', 'P_car', 'P_taxi', 'P_bike', 'P_PT']].idxmax(axis=1,
                                                                                                             skipna=True)

            if data.get('mode') == 1:
                fig_modal_sub = figure_modal_subs('Trip', data_before)
                fig_modal_sub_zone = figure_modal_subs('Zone', aggr_zone[0])

                map_src = generate_modal_subs_map(aggr_zone[0])
            else:
                fig_modal_sub = figure_modal_subs('Trip', data_before, data_after)
                # fig_modal_sub_zone = figure_modal_subs('Zone', aggr_zone[0], aggr_zone[1])

                aggr_zone[1]['modal_sub'] = aggr_zone[1][['P_walk', 'P_car', 'P_taxi', 'P_bike', 'P_PT']].idxmax(axis=1,
                                                                                                                 skipna=True)
                comps = aggr_zone[0][['geometry', 'id', 'modal_sub']].merge(aggr_zone[1][['id', 'modal_sub']],
                                                                            how='inner', on='id')
                comps['modal_sub'] = comps.apply(
                    lambda x: x['modal_sub_x'] if x['modal_sub_x'] == x['modal_sub_y'] else x['modal_sub_x'] + ' -> ' +
                                                                                            x['modal_sub_y'], axis=1)
                map_src = generate_modal_subs_map(comps)

            modal_card = dbc.Card(dbc.CardBody([
                html.H6("Modal Substitution Rate", className="card-title"),
                dcc.Graph(figure=fig_modal_sub, style={'height': '35vh'}),
                #dcc.Graph(figure=fig_modal_sub_zone, style={'height': '35vh'}),
            ], className="card-style", style={'height': '100%'}), className='card-container')

            div_map = dbc.Card([dbc.CardBody(
                [html.H6("Most likely substituted mode per zone", className="card-title"), dcc.Graph(figure=map_src)],
                className="card-style", style={'height': '100%'})], className='card-container',
                style={'height': '70vh'})

            return [], {'display': 'none'}, {'display': 'block'}, [
                dbc.Row([dbc.Col(modal_card, width=4), dbc.Col(div_map, width=8)])], []

        if len(aggr_zone) == 1:
            map_src = generate_single_map(aggr_zone[0], drop_val_name)
        else:
            map_src = generate_compare_map(aggr_zone[0], aggr_zone[1], drop_val_name)

        div_map = dbc.Card([dbc.CardBody([
            html.Iframe(id='spatial_map', srcDoc=map_src,
                        style={'width': '100%', 'height': '100%', 'overflow': 'hidden'})],
            className="card-style", style={'height': '100%'})], className='card-container', style={'height': '70vh'})

        if data.get('mode') == 1:
            div_descriptive = variable_descriptive_stats(drop_var, data.get('mode') - 1, data_before)
        else:
            div_descriptive = variable_descriptive_stats(drop_var, data.get('mode') - 1, data_before, data_after)

        # only for specific variables
        div_violin = html.Div()
        balance_card = html.Div()
        if drop_val_name in ['reduced_time', 'GHG']:
            if data.get('mode') == 1:
                violin_trip = dcc.Graph(id='violin-trip',
                                        figure=violin_plot_level('Trip', drop_var, data_before))
                violin_zone = dcc.Graph(id='violin-zone', figure=violin_plot_level('Zone', drop_var, aggr_zone[0]))
                if drop_val_name == 'GHG':
                    fig_balance = figure_balance_point(data_before)
            else:
                violin_trip = dcc.Graph(id='violin-trip-2',
                                        figure=violin_plot_level('Trip', drop_var, data_before, data_after))
                violin_zone = dcc.Graph(id='violin-zone-2',
                                        figure=violin_plot_level('Zone', drop_var, aggr_zone[0], aggr_zone[1]))
                if drop_val_name == 'GHG':
                    fig_balance = figure_balance_point(data_before, data_after)

            div_violin = dbc.Card([dbc.CardBody([dbc.Row([dbc.Col(violin_trip), dbc.Col(violin_zone)])],
                                                className="card-style", style={'height': '100%'})],
                                  className='card-container', style={'height': '70vh'})

            if drop_val_name == 'GHG':
                balance_card = dbc.Card(dbc.CardBody([
                    html.H6("Balance Points", className="card-title"),
                    dcc.Graph(figure=fig_balance, style={'height': '35vh'}),
                ], className="card-style", style={'height': '100%'}), className='card-container')

        if drop_var == 'Usage Efficiency':
            if tabs_val == 'filter-usage-eff':
                if data.get('mode') == 1:
                    df_ids = [get_ids_within_range(data_before, scooter_slide[0], scooter_slide[1])]
                    df_trips = [filter_data_via_ids(df_ids[0], data_before)]

                if data.get('mode') == 2:
                    df_ids = [get_ids_within_range(data_before, scooter_slide[0], scooter_slide[1])]
                    df_trips = [filter_data_via_ids(df_ids[0], data_before)]
                    df_ids.append(
                        get_ids_within_range(data_after, scooter_slide[0], scooter_slide[1]))
                    df_trips.append(filter_data_via_ids(df_ids[1], data_after))

                usage_freq_cont = usage_frequency_content(df_ids, df_trips, scooter_slide[0], scooter_slide[1])

                return [], {'display': 'none'}, {'display': 'block'}, [], [usage_freq_cont]

            return [], {'display': 'none'}, {'display': 'block'}, [
                dbc.Row([dbc.Col(div_descriptive, width=4), dbc.Col(div_map, width=8)])], []

        return [], {'display': 'none'}, {'display': 'block'}, [
            dbc.Row([dbc.Col(div_descriptive, width=4), dbc.Col(div_map, width=8)]), dbc.Row(div_violin), dbc.Row([balance_card])], []
