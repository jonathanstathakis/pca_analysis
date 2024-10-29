"""
Provide baseline correction insights.
"""

from dash import Input, Output, callback, dcc, html, register_page
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import Session

from pca_analysis.notebooks.experiments.parafac2_pipeline.database_etl_orm import Images
from pca_analysis.notebooks.experiments.parafac2_pipeline.app.database import engine
import logging

import dash_bootstrap_components as dbc

logger = logging.getLogger(__name__)

logging.basicConfig(level="DEBUG")

# allow standalone app
if not __name__ == "__main__":
    register_page(__name__)

layout = html.Div(
    [
        title := html.H1("Baseline Correction"),
        html.Div("provides baseline correction for a selected sample."),
        html.Br(),
        html.H2("select runid"),
        runid_selector := dcc.Dropdown(id="runid-selector"),
        html.Br(),
        html.H2("Signal"),
    ]
)


@callback(
    Output(runid_selector, "options"),
    Output(runid_selector, "value"),
    Input(title, "children"),
)
def get_runids(title):
    with engine.connect() as conn:
        result = conn.execute(text("select runid from inc_chm")).fetchall()
        conn.close()
    runids = [x[0] for x in result if x[0] not in ["82", "0151"]]

    return (
        runids,
        runids[0],
    )


if __name__ == "__main__":
    from dash import Dash

    app = Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO])
    app.layout = layout
    app.run(debug=True)
