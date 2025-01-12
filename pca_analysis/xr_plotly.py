"""
Adds a plotly accessor for xarray Datasets and DataArrays
"""

import xarray as xr
import plotly.express as px
from pca_analysis.xr_signal import plot_multiple_vars
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import cycle
import numpy as np


@xr.register_dataset_accessor("plotly")
class PlotlyAccessorDS:
    def __init__(self, ds):
        self._ds = ds

    def facet_plot_overlay(
        self,
        facet_col=None,
        var_keys=None,
        x_key=None,
        col_wrap: int = 1,
        fig_kwargs={},
        trace_kwargs: dict = {},
    ):
        """
        overlay multiple vars, assuming that after grouping by `grouper`, each var
        is 1D.
        """

        return plot_multiple_vars(
            ds=self._ds,
            facet_dim=facet_col,
            var_keys=var_keys,
            x_key=x_key,
            col_wrap=col_wrap,
            fig_kwargs=fig_kwargs,
            trace_kwargs=trace_kwargs,
        )


@xr.register_dataarray_accessor("plotly")
class PlotlyAccessorDA:
    def __init__(self, da):
        self._da = da

    def line(self, x, y=None, **kwargs):
        """
        Use `plotly_express` line plot api.

        Example
        -------

        ```
        fig = (da
            .plotly
            .line(x="mins", y="raw_data", color="sample")
            )
        ```

        """
        import plotly.express as px

        if y is None:
            y = self._da.name

        df = self._da.to_dataframe().reset_index()
        return px.line(df, x=x, y=y, **kwargs)

    def heatmap(self, x, y):
        return go.Figure(
            go.Heatmap(x=self._da[x], y=self._da[y], z=self._da.data.squeeze())
        )

    def facet(self, n_cols: int = 1, plot_type="line", x=None, y=None, z=None):
        return facet(da=self._da, n_cols=n_cols, plot_type=plot_type, x=x, y=y, z=z)


# iterate over sample, plotting each component as a trace.


def facet(da: xr.DataArray, n_cols=1, plot_type="heatmap", x=None, y=None, z=None):
    """
    Provides facet overlays of 3 dimensional data arrays. Useful when visualising
    chromatographic images who are too sparse and sharp for heatmaps to be informative.

    See `facet_plot_multiple_traces` for a Dataset equivalent.

    Hint: Use `update_layout` to make design changes after generation.

    TODO fix legend so only 1 entry per dim_2 line.
    """

    if plot_type not in ["line", "heatmap"]:
        raise ValueError
    if not isinstance(da, xr.DataArray):
        raise TypeError

    # NOTE leave space for other plot types downtrack.

    colormap = cycle(px.colors.qualitative.Plotly)

    if len(da.dims) > 3:
        raise ValueError("can only handle 3 dimensional arrays")

    if not z:
        z = da.dims[0]
    if not y:
        y = da.dims[1]
    if not x:
        x = da.dims[2]
    # generate the traces as key value pairs for each dim_1 subplot
    match plot_type:
        case "line":
            fig = multiple_lines(da=da, colormap=colormap, z=z, y=y, x=x, n_cols=n_cols)
        case "heatmap":
            fig = heatmaps(da=da, z=z, y=y, x=x, n_cols=n_cols)

    return fig


def multiple_lines(
    da,
    colormap,
    z,
    y,
    x,
    n_cols,
):
    traces = {}
    for z_label, z_da in da.transpose(z, ...).groupby(z):
        y_traces = {}
        for y_label, y_da in z_da.groupby(y):
            y_traces[str(y_label)] = go.Scatter(
                x=y_da[x],
                y=y_da.squeeze(),
                legendgroup=str(y_label),
                name=str(y_label),
                marker=dict(color=next(colormap)),
            )
        traces[str(z_label)] = y_traces

    fig = _build_line_subplots(traces=traces, n_cols=n_cols, x=x, da=da)

    return fig


def _build_line_subplots(
    traces: dict[str, go.Trace],
    n_cols: int,
    x: str,
    da: xr.DataArray,
):
    n_traces = len(traces)

    # determine number of rows as the rounding up of the ratio of traces to columns
    n_rows = int(np.ceil(n_traces / n_cols))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=list(traces.keys()),
        x_title=x,
        y_title=da.name,
    )

    curr_row = 1
    curr_col = 1
    for k, dim_2_traces in traces.items():
        for k, trace in dim_2_traces.items():
            fig.add_trace(trace, row=curr_row, col=curr_col)
        curr_col += 1
        if curr_col > n_cols:
            curr_col = 1
            curr_row += 1

    fig.update_layout(height=1000)

    return fig
