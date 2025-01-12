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

import plotly.io as pio

pio.templates.default = "seaborn"


@dataclass
class ChromDims:
    SAMPLE = "sample"
    TIME = "mins"
    SPECTRA = "wavelength"

    def __len__(self):
        return len(self.__dict__.keys())


@dataclass
class Names:
    """
    abstracted DataArray names such as chromatogram, baseline etc.
    """

    CHROM = "input_data"


chrom_dims = ChromDims()
names = Names()
CORE_DIM = chrom_dims.TIME


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


class VizCabernet(AbstChrom):
    def __init__(self, dt: DataTree):
        self._dt = dt

    def line_multiple_vars(
        self,
        vars: list[str],
        col_wrap: int = 1,
        x: str = "",
        facet_dim: str = "",
        color_dim: str = "",
        line_dash_dim: str = "",
    ):
        """
        compare the variables listed in `vars` visually. `vars` are `DataTree` paths to the variable.

        TODO unify plotting tools. Thus function should be able to
        handle all plotting cases including line or scatter etc.
        Merge with line.

        TODO modify so that can treat "vars" as just another dim
        like the dims. To do this will need to unpack according to
        the args order (x will contain the array of data, the
        remainder are the labels and the order of unpacking) then
        map the graphic effect to the dim - overlay, dash, color etc.
        Then compose the plot.
        """

        if not color_dim:
            color_dim = "vars"

        if not facet_dim:
            facet_dim = self.SAMPLE

        if not line_dash_dim:
            line_dash_dim = self.SPECTRA

        if not x:
            x = self.TIME

        if (
            (color_dim == facet_dim)
            or (color_dim == line_dash_dim)
            or (line_dash_dim == facet_dim)
        ):
            raise ValueError(
                f"need unique dim values, got {color_dim=}, {facet_dim=}, {line_dash_dim=}"
            )

        ds = self._dt.to_dataset()
        subplot_grps = ds.groupby(facet_dim)
        n_plots = len(subplot_grps.groups)
        n_rows = int(np.ceil(n_plots / col_wrap))

        fig_kwargs = {}
        fig_kwargs["x_title"] = x

        fig = make_subplots(
            rows=n_rows,
            cols=col_wrap,
            subplot_titles=list(str(x) for x in dict(subplot_grps).keys()),
        )

        curr_col = 1
        curr_row = 1

        # trace_kwargs_ = trace_kwargs.copy()

        import plotly.express as px

        colormap = px.colors.qualitative.Plotly[: len(vars)]

        line_dash_opts = [
            "solid",
            "dot",
            "dash",
            "longdash",
            "dashdot",
            "longdashdot",
        ] * len(ds.groupby(line_dash_dim).groups)

        for i, (k, g) in enumerate(subplot_grps):
            for v, c in zip(vars, colormap):
                for (kk, gg), ld in zip(g.groupby(line_dash_dim), line_dash_opts):
                    x_data = gg[x]
                    y_data = gg[v].squeeze()
                    trace = go.Scatter(
                        x=x_data,
                        y=y_data,
                        name=str(kk),
                        legendgroup=str(v),
                        legendgrouptitle=go.scatter.Legendgrouptitle(text=str(v)),
                        line=go.scatter.Line(color=c, dash=ld),
                        showlegend=False,
                    )
                    if i == 1:
                        trace.showlegend = True
                    fig.add_trace(trace, row=curr_row, col=curr_col)

            if curr_col == col_wrap:
                curr_col = 1
                curr_row += 1
            else:
                curr_col += 1
        return fig


class Cabernet(AbstChrom):
    def __init__(self, da: xr.DataArray):
        """
        Manager of data processes. Create a branched Tree of data transformations and
        mappings through a declarative(?) API utilising a XArray DataTree backend.

        TODO add peak picking
        TODO add smoothing
        TODO add sharpening
        TODO add clustering
        TODO add PARAFAC2

        TODO2 add sklearn pipeline
        TODO2 explore a polars style API for piping and constructing a branch at the
        same time. Need a method of avoiding having to write the parent path every
        time.
        """

        assert isinstance(da, DataArray), f"{type(da)}"

        self._dt = xr.DataTree(dataset=xr.Dataset({da.name: da}))

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

    def sel(self, **kwargs):
        self._dt = self._dt.sel(**kwargs)
        cab = self.copy()
        return cab

    def isel(self, **kwargs):
        self._dt = self._dt.isel(**kwargs)
        cab = self.copy()
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
        if items:
            if any(isinstance(x, Cabernet) for x in items.values()):
                items = {k: v._dt for k, v in items.items()}
        if items_kwargs:
            if any(isinstance(x, Cabernet) for x in items_kwargs.values()):
                items_kwargs = {k: v._dt for k, v in items_kwargs.items()}
        self._dt = self._dt.assign(items=items, **items_kwargs)

        cab = self.copy()

        return cab

    def __repr__(self):
        return repr(self._dt)

    def peak_picking(self, path, find_peaks_kwargs={}, peak_width_kwargs={}): ...

    def _repr_html_(self):
        return self._dt._repr_html_()

    def keys(self):
        return self._dt.keys()

    @property
    def data_vars(self):
        return self._dt.data_vars

    @property
    def attrs(self):
        return self._dt.attrs

    @property
    def name(self):
        return self._dt.name

    @property
    def sizes(self):
        return self._dt.sizes

    @property
    def viz(self):
        return VizCabernet(dt=self._dt)


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


class BCorr:
    def __init__(self, da: xr.DataArray):
        """
        baseline correction namespace.
        """

        self._da = da

    def snip(self, **kwargs):
        """
        TODO: convert to a scikit learn transformer if poss to reduce redundancy.
        """
        from pca_analysis.preprocessing.bcorr import snip as _snip

        ds = _snip(da=self._da, core_dim=CORE_DIM, **kwargs)

        dt = DataTree(ds)
        dt.name = "bcorred"
        dt.attrs["data_model_type"] = "baseline corrected"
        return Cabernet.from_tree(dt=dt)


class Transform:
    def __init__(self, da: xr.DataArray):
        """
        Shiraz transform operation namespace. transform the internal DataArray according
        to some predefined function, returning a Dataset or DataArray.
        """
        self._da = da
        self.bcorr = BCorr(da=self._da)


class Shiraz(AbstChrom):
    def __init__(self, da: xr.DataArray):
        """
        Manager for DataArrays
        TODO: add data model type specific transformer and visualiser classes and dispatcher.
        """

        assert isinstance(da, DataArray)

        self._da = da
        self.trans = Transform(da=self._da)

    def __len__(self):
        return self._da.__len__()

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
        self._da = self._da.sel(
            indexers=indexers,
            method=method,
            tolerance=tolerance,
            drop=drop,
            **indexers_kwargs,
        )
        shz = self.copy()
        return shz

    def isel(self, **kwargs):
        self._da = self._da.isel(**kwargs)
        shz = self.copy()
        return shz

    @property
    def shape(self):
        return self._da.shape

    @property
    def attrs(self):
        return self._da.attrs

    @property
    def viz(self) -> VizShiraz:
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

    def __repr__(self):
        return self._da.__repr__()

    def _repr_html_(self):
        return self._da._repr_html_()


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


def _line(da: xr.DataArray, x: str, trace_kwargs, fig_kwargs):
    assert da.ndim == 1, da.ndim
    fig = go.Figure(**fig_kwargs)
    trace = go.Scatter(x=da[x], y=da.data.squeeze(), **trace_kwargs)
    fig.add_trace(trace)

    return fig


def _heatmap(da: xr.DataArray, x: str, y: str, trace_kwargs, fig_kwargs):
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
