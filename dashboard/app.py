import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go


from components.callbacks import register_callbacks

pio.templates["fontmod"] = go.layout.Template(
    layout={
        'title':
            {'font': {'family': 'Open Sans, OpenSans, HelveticaNeue, Helvetica, Sans-serif'}
             },
        'font': {'family': 'Open Sans, OpenSans, HelveticaNeue, Helvetica, Sans-serif',
                 'color': 'dimgray',
                 'size': 10},
        'showlegend': False,
        'colorway': px.colors.qualitative.Pastel,
        'margin': {'l': 0, 'r': 0, 't': 60, 'b': 10}
    }
)
pio.templates.default = pio.templates["plotly_white+fontmod"]

app = dash.Dash(use_pages=True, external_stylesheets=['https://fonts.googleapis.com/css?family=Open+Sans:400,600,300',
                                                      dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='div-sidebar'),
    dash.page_container
])
register_callbacks(app)

if __name__ == '__main__':
    app.run(debug=True)
