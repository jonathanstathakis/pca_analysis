"""
Contains the experiment manager
"""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import xarray as xr
from plotly.subplots import make_subplots
from xarray import DataArray, DataTree, open_datatree

from pca_analysis.peak_picking_viz import plot_peaks

from . import AbstChrom
from . import chrom_dims
from ..get_dataset import get_shiraz_dataset
from .rank_estimation import RankEstimation
from .shiraz.shiraz import Shiraz
from pathlib import PurePath

pio.templates.default = "seaborn"
StrOrPath = str | PurePath


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

    def overlay_peaks(
        self,
        signal_path: StrOrPath,
        peaks_path: StrOrPath,
        group_dim: str = "",
        peak_outlines: bool = True,
        peak_width_calc: bool = True,
    ):
        """
        create a peak overlay of the peaks found at `peaks_path` and the signal at `signal_path`.


        Requires that the calling Cabernet has only 1 dimension: time.
        """
        ...

        from pca_analysis.peak_picking import get_peak_table_as_df

        peak_array = self._dt.get(str(peaks_path))

        assert isinstance(peak_array, DataArray)

        pdf = get_peak_table_as_df(pt=peak_array)

        signal_array = self._dt.get(str(signal_path))

        assert isinstance(signal_array, DataArray)

        fig = plot_peaks(
            ds=self._dt.to_dataset(),
            x=self.TIME,
            group_dim=group_dim,
            input_signal_key=signal_path,
            peak_table_key=peaks_path,
            peak_outlines=peak_outlines,
            peak_width_calc=peak_width_calc,
        )

        return fig


class Cabernet(AbstChrom):
    def __init__(self, da: DataArray):
        """
        Manager of data processes. Create a branched Tree of data transformations and
        mappings through a declarative(?) API utilising a XArray DataTree backend.

        TODO add smoothing
        TODO add sharpening
        TODO add clustering
        TODO add PARAFAC2

        TODO2 add sklearn pipeline
        TODO2 explore a polars style API for piping and constructing a branch at the same time. Need a method of avoiding having to write the parent path every time.
        """

        assert isinstance(da, DataArray), f"{type(da)}"

        self._dt = xr.DataTree(dataset=xr.Dataset({da.name: da}))

    @classmethod
    def from_file(
        cls,
        filename_or_obj,
        *,
        engine=None,
        chunks=None,
        cache=None,
        decode_cf=None,
        mask_and_scale=None,
        decode_times=None,
        decode_timedelta=None,
        use_cftime=None,
        concat_characters=None,
        decode_coords=None,
        drop_variables=None,
        inline_array=False,
        chunked_array_type=None,
        from_array_kwargs=None,
        backend_kwargs=None,
        **kwargs,
    ):
        dt = open_datatree(
            filename_or_obj,
            engine=engine,
            chunks=chunks,
            cache=cache,
            decode_cf=decode_cf,
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            decode_timedelta=decode_timedelta,
            use_cftime=use_cftime,
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            drop_variables=drop_variables,
            inline_array=inline_array,
            chunked_array_type=chunked_array_type,
            from_array_kwargs=from_array_kwargs,
            backend_kwargs=backend_kwargs,
            **kwargs,
        )

        cab = Cabernet.from_tree(dt=dt)

        return cab

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
        cab = self.copy()
        cab._dt = cab._dt.sel(**kwargs)
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

        if isinstance(result, DataTree):
            return Cabernet.from_tree(dt=result)
        elif isinstance(result, DataArray):
            return Shiraz(da=result)
        else:
            raise TypeError

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

    def peak_array_as_df(self, path: StrOrPath):
        from pca_analysis.peak_picking import get_peak_table_as_df

        peak_array = self._dt[str(path)]

        assert isinstance(peak_array, DataArray)
        df = get_peak_table_as_df(pt=peak_array)

        return df

    def pick_peaks(
        self, path: StrOrPath, find_peaks_kwargs={}, peak_width_kwargs={"rel_height": 1}
    ):
        """
        Map the peaks of the DataArray at `path`, adding the map as a DataArray at "{path} / 'peaks'", returning a copy
        Cabernet object with the peak map.
        """
        shz = self.get(path)

        assert isinstance(shz, Shiraz)

        peak_array = shz.pick_peaks(
            find_peaks_kwargs=find_peaks_kwargs, peak_width_kwargs=peak_width_kwargs
        )

        peak_path = PurePath(path).parent / "peaks"

        dt = self._dt.copy()

        dt[str(peak_path)] = peak_array

        cab = Cabernet.from_tree(dt=dt)

        return cab

    def _repr_html_(self):
        return self._dt._repr_html_()

    def keys(self):
        return self._dt.keys()

    def to_netcdf(
        self,
        filepath: StrOrPath,
        mode="w",
        encoding=None,
        unlimited_dims=None,
        format=None,
        engine=None,
        group=None,
        write_inherited_coords=False,
        compute=True,
        **kwargs,
    ):
        self._dt.to_netcdf(
            filepath=filepath,
            mode=mode,
            encoding=encoding,
            unlimited_dims=unlimited_dims,
            format=format,
            engine=engine,
            group=group,
            write_inherited_coords=write_inherited_coords,
            compute=compute,
            **kwargs,
        )

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

    @property
    def rank_estimation(self):
        return RankEstimation(dt=self._dt)

    @property
    def decomp(self):
        from .decomposition import Decomposition

        return Decomposition(dt=self._dt)

    @property
    def dims(self):
        return self._dt.dims
