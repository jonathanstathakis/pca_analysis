from pca_analysis import xr_plotly
from dash import Dash, dcc, Output, Input
from dash import Dash, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import xarray as xr


def chromviewer_2D(data: xr.Dataset):
    app = Dash(__name__)
    wavelength_vals = data.wavelength.values
    sample_vals = sorted(data.sample.values)
    default_wavelength = 256
    default_sample = 1

    # add sample selection.
    @app.callback(
        Output("figure1", "figure"),
        Input("wavelength-dropdown", "value"),
        Input("sample-dropdown", "value"),
    )
    def update_graph(wavelengths_selected, samples_selected):
        print(samples_selected)
        data_ = data.sel(sample=samples_selected)
        if isinstance(wavelengths_selected, list):
            if "all" not in wavelengths_selected:
                data_ = data.sel(wavelength=wavelengths_selected)
            else:
                data_ = data
        elif wavelengths_selected == "all":
            data_ = data
        else:
            data_ = data.sel(wavelength=wavelengths_selected)

        fig = go.Figure()
        for label, grp in data_.groupby(["sample", "wavelengths_selected"]):
            fig.add_trace(go.Scatter(x=grp["mins"], y=grp.data.squeeze(), label=label))

        fig = data_.plotly.line(
            x="mins", title=str(wavelengths_selected), color="wavelength"
        )
        return fig

    app.layout = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Dropdown(
                                id="wavelength-dropdown",
                                options=["all"] + list(wavelength_vals),
                                multi=True,
                                value=default_wavelength,
                            ),
                            dcc.Dropdown(
                                id="sample-dropdown",
                                options=["all"] + list(sample_vals),
                                multi=True,
                                value=default_sample,
                            ),
                        ]
                    ),
                    dbc.Col([dcc.Graph(id="figure1")]),
                ]
            )
        ]
    )

    return app
