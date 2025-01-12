"""
Contains the experiment manager
"""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import xarray as xr
from plotly.subplots import make_subplots
from xarray import DataArray, DataTree

from . import AbstChrom
from . import chrom_dims
from ..get_dataset import get_shiraz_dataset

from .shiraz.shiraz import Shiraz

pio.templates.default = "seaborn"


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
    def __init__(self, da: DataArray):
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

        cab = Cabernet(da=DataArray())
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

        cab = Cabernet(DataArray())

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
