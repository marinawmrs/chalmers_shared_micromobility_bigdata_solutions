from dash.dependencies import Output, Input

COL_ANT = ['#80b1d3', '#49516f']
COL_POST = ['#f89274', '#BA7872']
COL_SEC = ['#DCEED1']


def register_callbacks(app):
    """
    Handles callbacks that act in between components

    @param app: Dash app
    @return:
    """

    @app.callback(
        Output('div-sidebar', 'children'),
        [Input('url', 'pathname')]
    )
    def update_title(pathname):
        """
        Updates sidebar according to current page

        @param pathname: URL
        @return:
        """
        if pathname == '/dataset':
            from dashboard.pages import dataset
            return dataset.sidebar
        else:
            from dashboard.pages import dataset
            return dataset.sidebar



