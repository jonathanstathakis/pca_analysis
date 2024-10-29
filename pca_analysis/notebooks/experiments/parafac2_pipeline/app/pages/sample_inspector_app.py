import logging

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import plotly.express as px
import polars as pl
from dash import Input, Output, callback, dcc, html, register_page
from database_etl.etl.etl_pipeline_raw import (
    sample_metadata_query,
)
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import Session

from pca_analysis.notebooks.experiments.parafac2_pipeline.database_etl_orm import Images
from pca_analysis.notebooks.experiments.parafac2_pipeline.app.database import engine

logger = logging.getLogger(__name__)

logging.basicConfig(level="DEBUG")

register_page(__name__)
# app = Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])
# Base = declarative_base()

layout = html.Div(
    [
        title := dcc.Markdown("""
                              # Sample Inspector

                              A simple sample inspector app. A table of available samples
                              is provided. 2d and 3d visualisations are provided alongside
                              peak detection.
                              """),
        html.Br(),
        submit_exec_id := html.Button("submit"),
        selected_exec_id := html.Div(id="selected_exec_id"),
        html.Br(),
        metadata_header := dcc.Markdown("## Available Samples"),
        metadata := html.Div(),
        html.Br(),
        html.Br(),
        dcc.Markdown("## Select `runid`:"),
        runid_selector := dcc.Dropdown(id="si-runid-selector"),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Row(
                        [
                            dcc.Markdown("### Image"),
                            html.Br(),
                            html.H2("3D"),
                            wavelength_selector_3d := dcc.RangeSlider(
                                id="si-wavelength-selector-3d",
                                min=0,
                                max=1,
                            ),
                            mins_selector_3d := dcc.RangeSlider(
                                id="si-mins-selector-3d", min=0, max=1
                            ),
                            image_3d := dcc.Graph(id="plot-3d"),
                            html.H3("description"),
                            image_description := html.Div(),
                        ],
                        justify="center",
                    ),
                ),
                dbc.Col(
                    dbc.Row(
                        [
                            html.H3("Chromatogram 2D"),
                            wavelength_selector_2d := dcc.Dropdown(value=256),
                            mins_selector_2d := dcc.RangeSlider(
                                id="si-mins-selector-2d", min=0, max=1
                            ),
                            chromatogram_graph := dcc.Graph("chromatogram-graph"),
                            html.H3("Chromatogram Description"),
                            chromatogram_desc := dag.AgGrid(
                                className="ag-theme-alpine-dark",
                            ),
                        ],
                        justify="center",
                    )
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H3("Peaks"),
                                peak_table := dag.AgGrid(
                                    className="ag-theme-alpine-dark"
                                ),
                                peak_plot := dcc.Graph("peak-plot"),
                            ]
                        ),
                        dbc.Col(
                            [
                                html.H2("Param Table"),
                                findpeaks_param_tbl := dag.AgGrid(
                                    id="si-findpeaks-param-tbl",
                                    rowData=[
                                        {"parameter": "height", "value": None},
                                        {"parameter": "threshold", "value": None},
                                        {"parameter": "prominence", "value": 5},
                                        {"parameter": "width", "value": None},
                                        {"parameter": "wlen", "value": None},
                                        {"parameter": "rel_height", "value": 0.5},
                                        {"parameter": "plateau_size", "value": None},
                                    ],
                                    columnDefs=[
                                        {"field": "parameter"},
                                        {"field": "value", "editable": True},
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
            justify="center",
        ),
    ],
    style={
        "margin-left": "1rem",
        "margin-right": "1rem",
        "margin-top": "1rem",
        "margin-bottom": "1rem",
    },
)


@callback(Output(metadata, "children"), Input(title, "children"))
def display_metadata(title):
    df = pl.read_database(connection=engine, query=sample_metadata_query)

    # df = get_sample_metadata(c)
    return dag.AgGrid(
        id="si-sample-metadata",
        rowData=df.to_dicts(),
        columnDefs=[{"field": i} for i in df.columns],
        className="ag-theme-alpine-dark",
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


@callback(
    Output(wavelength_selector_3d, "min"),
    Output(wavelength_selector_3d, "max"),
    Output(wavelength_selector_3d, "step"),
    Output(wavelength_selector_3d, "value"),
    Input(runid_selector, "value"),
)
def set_wavelength_selector(runid):
    """for a given runid get the wavelength bounds to set the selector"""

    logger.debug(f"setting wavelength selector for {runid=}..")

    statement = select(func.min(Images.wavelength), func.max(Images.wavelength)).where(
        runid == runid
    )
    with Session(engine) as session:
        result = session.execute(statement).fetchall()

    min_nm = result[0][0]
    max_nm = result[0][1]
    step = 10
    val = [220, 270]

    logger.debug(f"returning wavelength range {min_nm=}, {max_nm=}, {step=}, {val=}.")
    return min_nm, max_nm, step, val


@callback(
    Output(mins_selector_3d, "min"),
    Output(mins_selector_3d, "max"),
    Output(mins_selector_3d, "step"),
    Output(mins_selector_3d, "value"),
    Input(runid_selector, "value"),
)
def set_mins_selector_3d(runid):
    """for a given runid get the wavelength bounds to set the selector"""

    logger.debug(f"setting wavelength selector for {runid=}..")

    statement = select(func.min(Images.mins), func.max(Images.mins)).where(
        runid == runid
    )
    with Session(engine) as session:
        result = session.execute(statement).fetchall()

    min_mins = result[0][0]
    max_mins = result[0][1]
    step = 10
    val = [min_mins, max_mins]

    logger.debug(f"returning mins range {min_mins=}, {max_mins=}, {step=}, {val=}.")
    return min_mins, max_mins, step, val


@callback(
    Output(image_3d, "figure"),
    Output(image_description, "children"),
    Input(runid_selector, "value"),
    Input(wavelength_selector_3d, "value"),
    Input(mins_selector_3d, "value"),
)
def get_image_3d(runid, wavelength, mins):
    # get the image data

    logger.debug(f"getting 3D data of {runid=} with {wavelength=}, {mins=}")

    statement = (
        select(Images)
        .where(
            Images.runid == runid,
            Images.wavelength.between(wavelength[0], wavelength[1]),
            Images.mins.between(mins[0], mins[1]),
        )
        .order_by(Images.runid, Images.wavelength, Images.mins)
    )

    # with Session(engine) as session:
    #     result = session.execute(statement).fetchall()

    stringified_statement = str(
        statement.compile(engine, compile_kwargs={"literal_binds": True})
    )
    logger.debug(stringified_statement)

    df = pl.read_database(query=stringified_statement, connection=engine)

    logger.debug(df.head())

    fig = px.line_3d(
        df,
        x=str(Images.wavelength.name),
        y=str(Images.mins.name),
        z=str(Images.abs.name),
        line_group=str(Images.wavelength.name),
    )

    img_desc = df.describe()

    return fig, dag.AgGrid(
        id="image-df-desc",
        rowData=img_desc.to_dicts(),
        columnDefs=[{"field": i} for i in img_desc.columns],
        className="ag-theme-alpine-dark",
    )


@callback(
    Output(mins_selector_2d, "min"),
    Output(mins_selector_2d, "max"),
    Output(mins_selector_2d, "step"),
    Output(mins_selector_2d, "value"),
    Input(runid_selector, "value"),
)
def set_mins_selector_2d(runid):
    """for a given runid get the wavelength bounds to set the selector"""

    logger.debug(f"setting wavelength selector for {runid=}..")

    statement = select(func.min(Images.mins), func.max(Images.mins)).where(
        runid == runid
    )
    with Session(engine) as session:
        result = session.execute(statement).fetchall()

    min_mins = result[0][0]
    max_mins = result[0][1]
    step = 5
    val = [min_mins, max_mins]

    logger.debug(f"returning mins range {min_mins=}, {max_mins=}, {step=}, {val=}.")
    return min_mins, max_mins, step, val


@callback(
    Output(wavelength_selector_2d, "options"),
    Output(wavelength_selector_2d, "value"),
    Input(runid_selector, "value"),
)
def set_wavelength_selector_2d(runid):
    logger.debug(f"setting wavelength selector for {runid=}..")

    statement = (
        select(Images.wavelength)
        .where(runid == runid)
        .distinct()
        .order_by(Images.wavelength)
    )

    with Session(engine) as session:
        result = session.execute(statement)
        wavelengths = [x[0] for x in result]

    middle_wavelength = wavelengths[len(wavelengths) // 2]
    logger.debug(f"setting wavelength: {middle_wavelength=}")

    return (wavelengths, middle_wavelength)


@callback(
    Output(chromatogram_graph, "figure"),
    Output(chromatogram_desc, "rowData"),
    Output(chromatogram_desc, "columnDefs"),
    Input(runid_selector, "value"),
    Input(wavelength_selector_2d, "value"),
    Input(mins_selector_2d, "value"),
)
def set_chromatogram_fig(runid, wavelength, mins):
    logger.debug("setting chromatogram fig..")
    logger.debug(f"getting 2D data of {runid=} with {wavelength=}, {mins=}")

    statement = (
        select(Images)
        .where(
            Images.runid == runid,
            Images.wavelength == wavelength,
            Images.mins.between(mins[0], mins[1]),
        )
        .order_by(Images.runid, Images.wavelength, Images.mins)
    )

    # with Session(engine) as session:
    #     result = session.execute(statement).fetchall()

    stringified_statement = str(
        statement.compile(engine, compile_kwargs={"literal_binds": True})
    )
    logger.debug(stringified_statement)

    df = pl.read_database(query=stringified_statement, connection=engine)

    logger.debug(df.head())

    fig = px.line(
        df,
        x=str(Images.mins.name),
        y=str(Images.abs.name),
    )

    cgram_desc = df.describe()

    rowdata = cgram_desc.to_dicts()
    columndefs = [{"field": i} for i in cgram_desc.columns]

    logger.debug("returning chromatogram data.")

    return (fig, rowdata, columndefs)


@callback(
    Output(
        peak_table,
        "rowData",
    ),
    Output(peak_table, "columnDefs"),
    Output(peak_plot, "figure"),
    Input(runid_selector, "value"),
    Input(wavelength_selector_2d, "value"),
    Input(mins_selector_2d, "value"),
    Input(findpeaks_param_tbl, "cellValueChanged"),
    Input(findpeaks_param_tbl, "rowData"),
)
def set_peak_table(runid, wavelength, mins, cell_val_changed, rowdata):
    from scipy.signal import find_peaks

    # for the initial callback firing the default cell values arnt passed in
    # cellValueChanged so use the initial rowData..
    logger.debug(
        f"input {cell_val_changed=}",
    )
    logger.debug(f"rowdata {rowdata=}")

    params = rowdata

    statement = (
        select(Images)
        .where(
            Images.runid == runid,
            Images.wavelength == wavelength,
            Images.mins.between(mins[0], mins[1]),
        )
        .order_by(Images.runid, Images.wavelength, Images.mins)
    )

    # with Session(engine) as session:
    #     result = session.execute(statement).fetchall()

    stringified_statement = str(
        statement.compile(engine, compile_kwargs={"literal_binds": True})
    )
    logger.debug(stringified_statement)

    df = pl.read_database(query=stringified_statement, connection=engine)

    flat_params = {}
    # validation. If is a digit, cast to float, else set to None. This wont throw errors
    #  if incorrect entries are provided..
    for x in params:
        if x["value"]:
            if isinstance(x["value"], str):
                if x["value"].isdigit():
                    flat_params[x["parameter"]] = float(x["value"])
                else:
                    pass
        else:
            flat_params[x["parameter"]] = None

    logger.debug(f"input params for find_peaks: {flat_params=}")
    peaks, _ = find_peaks(
        x=df["abs"].to_numpy(writable=True),
        **flat_params,
    )

    peak_table = df.with_row_index("idx").filter(pl.col("idx").is_in(peaks))

    # TODO: build peak table by indexing on df via the indexes in peaks, create a plot and output peak table, then try and do it so its a layer option on the 2d chromatogram.
    # TODO: spectral profile of selected peak.

    rowdata = peak_table.to_dicts()
    columndefs = [{"field": i} for i in peak_table.columns]

    logger.debug(f"peak table column names: {columndefs}")
    logger.debug(f"peak table row data: {rowdata[0:5]}")

    logger.debug(f"peak table: \n\n{peak_table.head()}")
    logger.debug(f"peak table shape: {peak_table.shape}")
    peak_plot = px.scatter(peak_table, x="mins", y="abs")
    return (rowdata, columndefs, peak_plot)


if __name__ == "__main__":
    from dash import Dash

    app = Dash(__name__)
    app.layout = layout
    app.run(debug=True)
