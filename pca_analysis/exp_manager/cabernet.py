"""
Contains the experiment manager
"""

import xarray as xr
from pca_analysis.get_dataset import get_shiraz_dataset
from xarray import Dataset, DataTree, DataArray
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Self


@dataclass
class ChromDims:
    SAMPLE = "sample"
    TIME = "mins"
    SPECTRA = "wavelength"

    def __len__(self):
        return len(self.__dict__.keys())


chrom_dims = ChromDims()


@dataclass
class Names:
    """
    abstracted DataArray names such as chromatogram, baseline etc.
    """

    CHROM = "input_data"


names = Names()


class AbstChrom:
    """
    provides a common *chromatogram* validation via the dims - a
    chromatogram should have `SAMPLE`, `TIME`, `SPECTRA` dims only.
    """

    CHROM = names.CHROM
    SAMPLE = chrom_dims.SAMPLE
    TIME = chrom_dims.TIME
    SPECTRA = chrom_dims.SPECTRA
    DIMS = [SAMPLE, TIME, SPECTRA]


class Cabernet(AbstChrom):
    @classmethod
    def from_tree(cls, dt: xr.DataTree):
        """
        Initilise Cabernet from a DataTree.
        """

        cab = Cabernet(da=xr.DataArray())
        cab._dt = dt

        return cab

    @classmethod
    def load_from_dataset(cls, name=None):
        if name == "shiraz":
            da = get_shiraz_dataset().input_data
        else:
            raise NotImplementedError

        return Cabernet(da=da)

    def __init__(self, da: xr.DataArray):
        """
        Manager of data processes
        """

        assert isinstance(da, DataArray), f"{type(da)}"

        self._dt = xr.DataTree(dataset=xr.Dataset({da.name: da}))

    def sel(self, **kwargs):
        cab = self.copy()
        cab = cab._dt.sel(**kwargs)
        return cab

    def isel(self, **kwargs):
        cab = self.copy()

        cab._dt = cab._dt.isel(**kwargs)
        return cab

    def __getitem__(self, key):
        result = self._dt[key]

        if isinstance(result, DataTree):
            return Cabernet.from_tree(dt=result)
        elif isinstance(result, DataArray):
            return Shiraz(da=result)
        else:
            return result

    def copy(self):
        """
        make a copy of the internal DataTree, returning a new Cabernet object.
        """

        dt = self._dt.copy()

        cab = Cabernet(xr.DataArray())

        cab._dt = dt

        return cab

    def __setitem__(self, key, data) -> None:
        self._dt[key] = data

    def get(self, key, default: DataTree | DataArray | None = None):
        result = self._dt.get(key=key, default=default)
        return result

    def assign(self, items=None, **items_kwargs):
        """
        Assign `DataTree` or `DataArray` to internal `DataTree`.
        """
        cab = self.copy()
        cab._dt = cab._dt.copy().assign(items=items, **items_kwargs)

        return cab

    def __repr__(self):
        return repr(self._dt)

    def _repr_html_(self):
        return self._dt._repr_html_()


class PinotNoir(AbstChrom):
    def __init__(self, ds: Dataset):
        """
        Manager for Datasets.
        """

        assert isinstance(ds, Dataset)

        self._ds = ds

    def __getitem__(self, key):
        result = self._ds[key]

        if isinstance(result, DataTree):
            return
        elif isinstance(result, DataArray):
            return Shiraz(da=result)
        else:
            return result

    def __setitem__(self, key, data):
        self._ds[key] = data

    def get(self, key, default=None):
        result = self._ds.default(name=key, default=default)

        if isinstance(result, xr.Dataset):
            return PinotNoir(ds=result)
        elif isinstance(result, xr.DataArray):
            return Shiraz(da=result)
        else:
            return result


class Shiraz(AbstChrom):
    def __init__(self, da: xr.DataArray):
        """
        Manager for DataArrays
        """

        assert isinstance(da, DataArray)

        self._da = da

    def copy(self):
        """
        make a copy of the internal DataTree, returning a new Cabernet object.
        """

        da = self._da.copy()

        shz = Shiraz(da=da)

        return shz

    def sel(
        self, indexers=None, method=None, tolerance=None, drop=False, **indexers_kwargs
    ):
        shz = self.copy()
        shz._da = shz._da.sel(
            indexers=indexers,
            method=method,
            tolerance=tolerance,
            drop=drop,
            **indexers_kwargs,
        )
        return shz

    def isel(self, **kwargs):
        shz = self.copy()

        shz._da = shz._da.isel(**kwargs)
        return shz

    @property
    def viz(self):
        return VizShiraz(da=self._da)

    def __getitem__(self, key):
        result = self._da[key]

        if isinstance(result, Dataset):
            return Shiraz(da=result)
        else:
            return result

    def __setitem__(self, key, data):
        self._da[key] = data

        return Shiraz(da=self._da)

    def get(self, key, default=None):
        result = self._da.__get__(name=key, default=default)
        return result


class VizShiraz(AbstChrom):
    x = 1

    def __init__(self, da: DataArray):
        assert isinstance(da, DataArray)
        self._da = da

    def heatmap(self, n_cols: int = 1, trace_kwargs={}, fig_kwargs={}):
        if self._da[self.SAMPLE].data.ndim == 0:
            fig = _heatmap(
                da=self._da,
                x=self.SPECTRA,
                y=self.TIME,
                trace_kwargs=trace_kwargs,
                fig_kwargs=fig_kwargs,
            )

            fig.update_layout(
                title=dict(text=f"{self.SAMPLE}: {self._da[self.SAMPLE].data}")
            )

        else:
            fig = _facet_heatmap(
                da=self._da,
                x=self.SPECTRA,
                y=self.TIME,
                facet_dim=self.SAMPLE,
                trace_kwargs=trace_kwargs,
                fig_kwargs=fig_kwargs,
                n_cols=n_cols,
            )

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

            fig.show()
        else:
            raise ValueError("can only handle 1, 2 or 3 dims.")
        return fig


def _line_facet(
    da: xr.DataArray,
    x: str,
    overlay_dim: str,
    facet_dim: str,
    trace_kwargs: dict = {},
    fig_kwargs: dict = {},
    n_cols: int = 1,
):
    """
    Line faceting where one dim is displayed as the faceting, the second as plot overlays
    and a third as the x axis of each plot.

    TODO sort subplots lexographically or numerically according to label
    TODO fix legend grouping to display 1 per `overlay_dim` value. We know this works.
    See <https://plotly.com/python/legend/> subheadings "Grouped Legend Items"
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
                showlegend=True,
                **trace_kwargs,
            )
        traces[str(facet_label)] = overlay_traces

    fig = _build_line_subplots_fig(
        traces=traces, n_cols=n_cols, x=x, da=da, fig_kwargs=fig_kwargs
    )

    return fig


def _build_line_subplots_fig(
    traces: dict[str, go.Trace],
    n_cols: int,
    x: str,
    da: xr.DataArray,
    fig_kwargs: dict = {},
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

    fig.update_layout(height=1000)

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


def _line(da: xr.DataArray, x: str, trace_kwargs, fig_kwargs):
    assert da.ndim == 1, da.ndim
    fig = go.Figure(**fig_kwargs)
    trace = go.Scatter(x=da[x], y=da.data.squeeze(), **trace_kwargs)
    fig.add_trace(trace)

    return fig


def _heatmap(da: xr.DataArray, x: str, y: str, trace_kwargs, fig_kwargs):
    fig = go.Figure(**fig_kwargs)
    trace = go.Heatmap(x=da[x], y=da[y], z=da.data.squeeze(), **trace_kwargs)
    fig.add_trace(trace)
    return fig


def _facet_heatmap(
    da: xr.DataArray,
    x: str,
    y: str,
    facet_dim: str,
    trace_kwargs: dict = {},
    fig_kwargs: dict = {},
    n_cols: int = 1,
):
    """
    TODO: fix multiple generation of color scale bar.
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
                x=dim_1_da[x],
                y=dim_1_da[y],
                z=dim_1_da.data.squeeze(),
                name=str(dim_1_label),
                **trace_kwargs,
            ),
            row=curr_row,
            col=curr_col,
        )
        curr_col += 1
        if curr_col > n_cols:
            curr_col = 1
            curr_row += 1

    fig.update_layout(height=1000, title=f"heatmaps of {da.name} over '{facet_dim}'")

    return fig
