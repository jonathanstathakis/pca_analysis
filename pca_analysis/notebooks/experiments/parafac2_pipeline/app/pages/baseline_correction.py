"""
Provide baseline correction insights. A proof of concept.
"""

from dash import Input, Output, callback, dcc, html, register_page
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import Session

from pca_analysis.notebooks.experiments.parafac2_pipeline.database_etl_orm import Images
from pca_analysis.notebooks.experiments.parafac2_pipeline.app.database import engine
import logging

import dash_bootstrap_components as dbc
import polars as pl
from pybaselines import Baseline
import plotly.express as px

logger = logging.getLogger(__name__)

logging.basicConfig(level="DEBUG")

# allow standalone app
if not __name__ == "__main__":
    register_page(__name__)

layout = html.Div(
    [
        html.H1(id="bcorr-title", children="Baseline Correction"),
        html.Div("provides baseline correction for a selected sample."),
        html.Br(),
        html.H2("select runid"),
        dcc.Dropdown(id="bcorr-runid-selector"),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("select wavelength"),
                        dcc.Dropdown(id="bcorr-wavelength-selector", multi=False),
                        html.Br(),
                    ],
                ),
                dbc.Col(
                    [
                        html.H2("Signal"),
                        dcc.Graph(id="bcorr-graph"),
                    ]
                ),
            ]
        ),
    ]
)


@callback(
    Output("bcorr-runid-selector", "options"),
    Output("bcorr-runid-selector", "value"),
    Input("bcorr-title", "children"),
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


@callback(
    Output("bcorr-wavelength-selector", "options"),
    Output("bcorr-wavelength-selector", "value"),
    Input("bcorr-runid-selector", "value"),
)
def get_bcorr_wavelength_selector_params(runid):
    wavelengths = [
        x[0]
        for x in Session(engine).execute(
            select(Images.wavelength)
            .where(Images.runid == runid)
            .distinct()
            .order_by(Images.wavelength)
        )
    ]

    return wavelengths, wavelengths[len(wavelengths) // 2]


@callback(
    Output("bcorr-graph", "figure"),
    Input("bcorr-runid-selector", "value"),
    Input("bcorr-wavelength-selector", "value"),
)
def get_bcorr_graph(runid, wavelength):
    """
    run baseline correction and generate a plot
    """

    logger.debug(msg=f"wavelength input: {wavelength=}")

    # get the data
    stmt = (
        select(Images)
        .where(Images.runid == runid, Images.wavelength == wavelength)
        .order_by(Images.runid, Images.wavelength, Images.mins)
    )

    df = pl.read_database(
        str(stmt.compile(engine, compile_kwargs={"literal_binds": True})), engine
    )

    bcorrs = []
    weights = []
    tol_history = []

    for x in df.partition_by(Images.wavelength.name):
        add_wavelength_col = pl.lit(x.get_column(Images.wavelength.name)[0]).alias(
            Images.wavelength.name
        )
        data = x.get_column(Images.abs.name)
        baseline = Baseline()

        bcorr, params = baseline.asls(data=data)

        bcorr_df_ = (
            pl.DataFrame({"baseline": bcorr})
            .with_columns(add_wavelength_col)
            .with_row_index("idx")
        )

        bcorrs.append(bcorr_df_)

        weight_df_ = (
            pl.DataFrame({"weight": params["weights"]})
            .with_columns(add_wavelength_col)
            .with_row_index("idx")
        )

        weights.append(weight_df_)

        tol_history_df_ = (
            pl.DataFrame({"tol_history": params["tol_history"]})
            .with_columns(add_wavelength_col)
            .with_row_index("idx")
        )

        tol_history.append(tol_history_df_)

    bcorr_df = pl.concat(bcorrs)
    weight_df = pl.concat(weights)
    tol_history_df = pl.concat(tol_history)

    # todo: add index col to images table

    df = (
        df.rename({Images.abs.name: "input"})
        .with_columns(
            pl.col(Images.mins.name)
            .rank("dense")
            .over(Images.wavelength.name)
            .alias("idx")
        )
        .join(bcorr_df, on=[Images.wavelength.name, "idx"])
        .unpivot(
            index=[Images.runid.name, Images.wavelength.name, "idx", Images.mins.name],
            variable_name="signal",
            value_name=Images.abs.name,
        )
    )
    logger.debug(df.head())

    # fig = px.line(df, x='mins')

    # execute the correction

    # provide the result

    return px.line(
        df,
        x=Images.mins.name,
        y=Images.abs.name,
        color="signal",
        line_dash="signal",
    )


if __name__ == "__main__":
    from dash import Dash

    app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
    app.layout = layout
    app.run(debug=True)
