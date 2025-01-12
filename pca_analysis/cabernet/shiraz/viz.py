from pca_analysis.cabernet import (
    AbstChrom,
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from xarray import DataArray


class VizShiraz(AbstChrom):
    def __init__(self, da: DataArray):
        assert isinstance(da, DataArray)
        self._da = da

    def heatmap(self, n_cols: int = 1, trace_kwargs={}, fig_kwargs={}):
        da_reshaped = self._da.transpose(self.TIME, self.SPECTRA, ...)

        if da_reshaped.data.ndim == 2:
            fig = _heatmap(
                da=da_reshaped,
                x=self.SPECTRA,
                y=self.TIME,
                trace_kwargs=trace_kwargs,
                fig_kwargs=fig_kwargs,
            )

            fig.update_layout(
                title=dict(text=f"{self.SAMPLE}: {da_reshaped[self.SAMPLE].data}")
            )

        elif da_reshaped.data.ndim == 3:
            fig = _heatmap_facet(
                da=da_reshaped,
                x=self.SPECTRA,
                y=self.TIME,
                facet_dim=self.SAMPLE,
                trace_kwargs=trace_kwargs,
                fig_kwargs=fig_kwargs,
                n_cols=n_cols,
            )
        else:
            raise ValueError("can only handle 2 or 3 dim data")

        return fig

    def line(
        self,
        x: str = "",
        n_cols: int = 1,
        facet_dim: str = "",
        overlay_dim: str = "",
        trace_kwargs={},
        fig_kwargs={},
    ):
        """
        If x is time dim, produce a chromatogram, else if spectra, produce a spectrogram
        Need three points of logic here. If a 3 dimensional DataArray is passed then
        we need to iterate through the options. 3 dims means we generate a facet
        of overlays, where the facet col becomes dim 1, overlay col is dim 2, and dim 3
        (time) is the x axis.

        The first is to dispatch the scenario where a 1 dim array is passed, i.e.
        we've selected a sample and a wavelength.
        """

        if x is None:
            raise ValueError(f"select {self.TIME} or {self.SPECTRA}")

        assert isinstance(x, str)
        assert isinstance(facet_dim, str)
        assert isinstance(trace_kwargs, dict)
        assert isinstance(fig_kwargs, dict)

        if self._da.ndim == 1:
            fig = _line(
                da=self._da,
                x=x,
                trace_kwargs=trace_kwargs,
                fig_kwargs=fig_kwargs,
            )

            if x == self.TIME:
                title_text = f"{self.SAMPLE}: {self._da[self.SAMPLE].data}, {self.SPECTRA}: {self._da[self.SPECTRA].data}"
            else:
                title_text = f"{self.SAMPLE}: {self._da[self.SAMPLE].data}, {self.TIME}: {self._da[self.TIME].data}"

            fig.update_layout(
                title=dict(text=title_text),
                xaxis=dict(title=dict(text=x)),
                yaxis=dict(title=dict(text=self._da.name)),
            )

        elif self._da.ndim == 2:
            fig = _line_overlay(
                da=self._da,
                x=x,
                overlay_dim=overlay_dim,
                trace_kwargs=trace_kwargs,
                fig_kwargs=fig_kwargs,
            )

            # title_text = f"{}"
            scalar_dim_set = set(self.DIMS).difference([x, overlay_dim])
            if len(scalar_dim_set) > 1:
                raise ValueError
            else:
                scalar_dim = list(scalar_dim_set)[0]
            fig.update_layout(
                title=dict(
                    text=f"{scalar_dim}: {self._da[scalar_dim].data}, overlaid by {overlay_dim}"
                ),
                xaxis=dict(title=dict(text=x)),
                yaxis=dict(title=dict(text=self._da.name)),
            )

        elif self._da.ndim == 3:
            fig = _line_facet(
                da=self._da,
                x=x,
                overlay_dim=overlay_dim,
                facet_dim=facet_dim,
                trace_kwargs=trace_kwargs,
                fig_kwargs=fig_kwargs,
                n_cols=n_cols,
            )
        else:
            raise ValueError("can only handle 1, 2 or 3 dims.")
        return fig


def _line_facet(
    da: DataArray,
    x: str,
    overlay_dim: str,
    facet_dim: str,
    trace_kwargs: dict = {},
    fig_kwargs: dict = {},
    n_cols: int = 1,
):
    """
    Line faceting where one dim is displayed as the faceting, the second as plot overlays and a third as the x axis of each plot.

    TODO sort subplots lexographically or numerically according to label
    """
    from itertools import cycle
    import plotly.express.colors as colors

    traces = {}
    for facet_label, facet_da in da.transpose(facet_dim, ...).groupby(facet_dim):
        overlay_traces = {}
        colormap = cycle(colors.qualitative.Plotly)
        for overlay_label, overlay_da in facet_da.groupby(overlay_dim):
            overlay_traces[str(overlay_label)] = go.Scatter(
                x=overlay_da[x],
                y=overlay_da.squeeze(),
                legendgroup=str(overlay_label),
                name=str(overlay_label),
                marker=dict(color=next(colormap)),
                showlegend=False,
                **trace_kwargs,
            )
        traces[str(facet_label)] = overlay_traces

    n_traces = len(traces)

    # determine number of rows as the rounding up of the ratio of traces to columns
    n_rows = int(np.ceil(n_traces / n_cols))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=list(traces.keys()),
        x_title=x,
        y_title=da.name,
        **fig_kwargs,
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

    # only display legend of first plot.
    fig.update_traces(patch=dict(showlegend=True), row=1, col=1)

    fig.update_layout(height=1000, legend_title_text=str(overlay_dim))
    return fig


def _line_overlay(
    da: DataArray, x: str, overlay_dim: str, trace_kwargs={}, fig_kwargs={}
):
    assert x
    assert overlay_dim
    assert x != overlay_dim, "x must be different from overlay_dim"
    assert isinstance(trace_kwargs, dict)
    assert isinstance(fig_kwargs, dict)

    fig = go.Figure(**fig_kwargs)

    for k, grp in da.groupby(overlay_dim):
        fig.add_trace(
            go.Scatter(
                x=grp[x],
                y=grp.data.squeeze(),
                legendgroup=overlay_dim,
                legendgrouptitle=dict(text=str(overlay_dim)),
                name=str(k),
                **trace_kwargs,
            )
        )

    return fig


def _line(da: DataArray, x: str, trace_kwargs, fig_kwargs):
    assert da.ndim == 1, da.ndim
    fig = go.Figure(**fig_kwargs)
    trace = go.Scatter(x=da[x], y=da.data.squeeze(), **trace_kwargs)
    fig.add_trace(trace)

    return fig


def _heatmap(da: DataArray, x: str, y: str, trace_kwargs, fig_kwargs):
    """
    # TODO add x and y labels.
    """
    fig = go.Figure(**fig_kwargs)
    trace = go.Heatmap(x=da[x], y=da[y], z=da.data.squeeze(), **trace_kwargs)
    fig.add_trace(trace)
    fig.update_layout(
        xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text=str(x))),
        yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text=str(y))),
    )
    return fig


def _heatmap_facet(
    da: DataArray,
    x: str,
    y: str,
    facet_dim: str,
    trace_kwargs: dict = {},
    fig_kwargs: dict = {},
    n_cols: int = 1,
):
    """
    TODO: fix multiple generation of color scale bar.
    TODO: integrate datashader <https://plotly.com/python/datashader/> for
    handling large datasets.
    """
    grpby = da.groupby(facet_dim)
    n_traces = len(grpby.groups)
    n_rows = int(np.ceil(n_traces / n_cols))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[str(x) for x in grpby.groups.keys()],
        x_title=x,
        y_title=y,
        **fig_kwargs,
    )
    curr_row = 1
    curr_col = 1
    for dim_1_label, dim_1_da in grpby:
        fig.add_trace(
            trace=go.Heatmap(
                x=dim_1_da[x].values,
                y=dim_1_da[y].values,
                z=dim_1_da.data.squeeze(),
                name=str(dim_1_label),
                coloraxis="coloraxis",
                **trace_kwargs,
            ),
            row=curr_row,
            col=curr_col,
        )
        curr_col += 1
        if curr_col > n_cols:
            curr_col = 1
            curr_row += 1

    fig.update_layout(
        height=1000,
        title=f"heatmaps of {da.name} over '{facet_dim}'",
        coloraxis=go.layout.Coloraxis(),
    )

    return fig
