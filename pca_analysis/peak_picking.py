"""
In chromatography, a peak table generally consists of the time location of the maxima, the left and right bounds, the width and the area.

`scipy.signal` provides functions for obtaining the location, height, width, prominence, left and right base of detected peaks. Other parameters can be derived from these fundamental ones.
"""

import xarray as xr
from . import xr_signal
import plotly.graph_objects as go
from itertools import cycle
from scipy import signal
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.express as px
from dataclasses import dataclass


class Grid:
    def __init__(self, n_cells: int, col_wrap: int):
        """
        compute a mapping between an array of subplots and a subplot grid
        """

        self.n_cells = n_cells
        self.col_wrap = col_wrap
        self.n_rows = int(np.ceil(n_cells / col_wrap))
        self.cells = []
        self.construct_grid()

    def construct_grid(self):
        """
        build a grid coordinate structure as a list of tuples
        """

        for row in range(1, self.n_rows + 1):
            for col in range(1, self.col_wrap + 1):
                self._add_cell(row=row, col=col)

    def _add_cell(self, row: int, col: int):
        self.cells.append(Cell(row=row, col=col))

    def display_grid(self):
        for row_idx in range(self.n_rows + 1):
            row = [x for x in self.cells if x[0] == row_idx]
            print(row)

    def map_flat_grid(self):
        return {k: v for k, v in enumerate(self.cells)}


class Cell:
    def __init__(self, row, col):
        self.row = row
        self.col = col


from plotly.basedatatypes import BaseTraceType
from typing import Any


class TraceCollection:
    def __init__(self, traces: dict[Any, BaseTraceType] = {}, name: str = ""):
        """
        For treating collections of traces as single object

        Parameters
        ==========

        traces: dict[str, BaseTraceType]
            a labelled collection of Plotly traces, i.e. go.Scatter
        name: str = ""
            the name of the collection

        Methods
        =======

        """

        assert isinstance(traces, dict)
        assert isinstance(name, str)

        self._traces = traces
        self.name = name

    def __getitem__(self, key):
        return self._traces[key]

    def __setitem__(self, key, data):
        self._traces[key] = data

    def __repr__(self):
        import pprint

        repr_dict = {}

        for k, v in self._traces.items():
            repr_dict[k] = dict(name=v.name, type=type(v), meta=v["meta"])
        repr_str = (
            f"""TraceCollection\n"""
            """===============\n"""
            f"""len: {len(self.traces)}\n"""
            f"""{pprint.pformat(repr_dict)}\n"""
        )
        return repr_str

    @property
    def traces(self):
        return list(self._traces.values())


# TODO define a Trace collectoin type for handling collections of related traces as single objects. Main point is to define view_setting, update_setting methods on it to control how it displays.


class SubPlot:
    """
    collection of traces and other data for each subplot
    """

    def __init__(
        self,
        peak_table: pd.DataFrame = pd.DataFrame(),
        input_signal_data=None,
        sample: str = "",
        title: str = "",
    ):
        self.input_signal_data = input_signal_data
        self.peak_table = peak_table
        self.sample = sample
        self.title = title

        self.signal = go.Scatter()
        self.peak_width_calcs = TraceCollection(name="peak_width_calcs")
        self.maximas = TraceCollection(name="maximas")
        self.peak_outlines = TraceCollection(name="peak_outlines")

        self._attr_names = [
            "signal",
            "peak_width_calcs",
            "maximas",
            "peak_outlines",
        ]

        colormap = px.colors.qualitative.Plotly
        self.signal_color = colormap[0]
        peak_colors = colormap[1:]

        if len(peak_colors) < self.peak_table.shape[0]:
            diff = self.peak_table.shape[0] - len(peak_colors)
            self.peak_colors = peak_colors + peak_colors[:diff]
        else:
            self.peak_colors = colormap

    def update_attr(self, name: str, **kwargs):
        if name == "all":
            for name in self._attr_names:
                self._update_attr(name, **kwargs)
        else:
            self._update_attr(name=name, **kwargs)

    def _view_setting(self, settings: list[str]):
        """
        provide a view of all attribute settings
        """

        settings_mapping = {}

        # need to iterate over the dict of traces or
        # the signal trace equally.

        # for k in self._attr_names:
        #     for setting in settings:
        #         attr = setting_val = getattr(self, k)

        #         if isinstance(attr, go.Scatter):
        #             setting_val =
        #         if isinstance(attr, dict):
        #             attr_settings_dict = {}
        #             for k, v in attr.items():
        #                 v[]

        #         settings_mapping[k] = {}
        #         settings_mapping[k][setting] = setting_val

        return settings_mapping

    def _update_attr(self, name: str, **kwargs):
        """
        update attribute
        """
        attr = getattr(self, name)

        if isinstance(attr, dict):
            new_attr = {}
            for k, v in attr.items():
                if not isinstance(v, go.Scatter):
                    raise TypeError("expected Scatter object")

                new_v = v.update(**kwargs)

                new_attr[k] = new_v
            setattr(self, name, new_attr)

        elif isinstance(attr, go.Scatter):
            new_attr = attr.update(**kwargs)
            setattr(self, name, new_attr)
        else:
            raise TypeError(f"expected dict or Scatter. Cant update {type(attr)}")

    def __str__(self):
        return f"""
        SubPlot
        =======
        
        sample: {self.sample}
        title: {self.title}
        """

    def __repr__(self):
        return str(self)

    def _trace_signal(self):
        self.signal = go.Scatter(
            y=self.input_signal_data,
            name="signal",
            showlegend=False,
            meta=dict(sample=self.sample),
            legendgroup="signal",
            line_color=self.signal_color,
        )

    def _trace_outlines(self):
        for color, (idx, row) in zip(self.peak_colors, self.peak_table.iterrows()):
            self.peak_outlines[idx] = go.Scatter(
                x=[row["left_ip"], row["p_idx"], row["right_ip"]],
                y=[row["width_height"], row["maxima"], row["width_height"]],
                name="outline",
                mode="lines",
                line_color=color,
                line_width=1,
                legendgroup="outline",
                showlegend=False,
                meta=dict(peak=idx, sample=self.sample),
            )

    def _trace_peak_width_calc(self):
        for color, (idx, row) in zip(self.peak_colors, self.peak_table.iterrows()):
            self.peak_width_calcs[idx] = go.Scatter(
                x=[
                    row["left_ip"],
                    row["right_ip"],
                    None,
                    row["p_idx"],
                    row["p_idx"],
                ],
                y=[
                    row["width_height"],
                    row["width_height"],
                    None,
                    row["maxima"],
                    row["width_height"],
                ],
                name="width",
                mode="lines",
                line_dash="dot",
                line_width=0.75,
                line_color=color,
                legendgroup="width_calc",
                showlegend=False,
                meta=dict(peak=idx, sample=self.sample),
                customdata=[],
            )

    def _trace_maxima(self):
        for color, (idx, row) in zip(self.peak_colors, self.peak_table.iterrows()):
            self.maximas[idx] = go.Scatter(
                x=[row["p_idx"]],
                y=[row["maxima"]],
                mode="markers",
                name="maxima",
                marker_color=color,
                marker_size=5,
                legendgroup="maxima",
                showlegend=False,
                meta=dict(peak=idx, sample=self.sample),
            )

    def draw_traces(self):
        self._trace_signal()
        self._trace_peak_width_calc()
        self._trace_peak_width_calc()
        self._trace_maxima()

        assert True

    def __getitem__(self, key):
        return getattr(self, key)

    @property
    def traces(self) -> list:
        traces = []

        if self.signal:
            traces.append(self.signal)
        for attr in [
            self.peak_outlines,
            self.peak_width_calcs,
            self.maximas,
        ]:
            if attr:
                traces.extend(attr.traces)

        return traces

    def preview_plot(self):
        """
        return a figure object of the internal traces
        """

        traces = self.traces

        fig = go.Figure()

        fig.add_traces(traces)

        return fig


@dataclass
class SubPlotCollection:
    """
    provide methods to return information about internal Subplopts
    """

    subplots: list[SubPlot]

    @property
    def titles(self):
        return [str(x.title) for x in self.subplots]

    def __len__(self):
        return len(self.subplots)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.subplots[key]

    def __setitem__(self, key, data):
        if isinstance(key, int):
            self.subplots[key] = data

    def view_setting(self, setting: list[str]):
        settings = {}
        for idx, subplot in enumerate(self.subplots):
            settings[idx] = subplot._view_setting(settings=setting)

    def normalize_legend(self):
        """
        Set the first subplot showlegend = True, others False. Use when markers across
        subplots belong to the same class.
        """

        subplot_0 = self.subplots[0]
        subplot_0.update_attr(name="all", showlegend=True)

        self.subplots[0] = subplot_0

        for idx, subplot in enumerate(self.subplots[1:]):
            subplot.update_attr("all", showlegend=False)

            self.subplots[idx + 1] = subplot

    def sel(self, **kwargs):
        """
        select subplots matching kwargs.
        """

        for key in kwargs.keys():
            if not hasattr(self.subplots[0], key):
                raise ValueError(f"{key} not a valid accessor")

        temp = []

        for key, val in kwargs.items():
            for subplot in self.subplots:
                if subplot[key] == val:
                    temp.append(subplot)

        return temp

    def __iter__(self):
        return iter(self.subplots)


PEAKS = "peaks"


def _get_da_as_df(da: xr.DataArray):
    """return the dataarray as a dataframe with only the key dimensional variables and the data."""
    drop_vars = set(da.coords).difference(set(da.sizes.keys()))

    return da.drop_vars(drop_vars).to_dataframe()


def get_peak_table_as_df(pt: xr.DataArray):
    """
    return the peak table as a flattened, pivoted dataframe
    """

    pt_ = _get_da_as_df(da=pt).unstack().droplevel(0, axis=1).dropna().reset_index()
    return pt_


def find_peaks(
    ds: xr.Dataset,
    signal_key: str,
    find_peaks_kws: dict,
    grouper: list[str],
    x_key: str,
    maxima_coord_name: str = PEAKS,
    by_maxima: bool = True,
):
    if not by_maxima:
        input_signal_key = signal_key
        analysed_signal_key = "{signal_key}_inverted"
        ds = ds.assign(**{analysed_signal_key: lambda x: x[input_signal_key] * -1})
    else:
        analysed_signal_key = signal_key

    ds_ = ds.pipe(
        xr_signal.find_peaks_dataset,
        array_key=analysed_signal_key,
        grouper=grouper,
        new_arr_name=maxima_coord_name,
        x_key=x_key,
        find_peaks_kws=find_peaks_kws,
    )

    return ds_


def tablulate_peaks_1D(
    x, find_peaks_kwargs=dict(), peak_widths_kwargs=dict(rel_height=0.95)
):
    """
    Generate a peak table for a 1D input signal x.

    TODO4 test

    TODO decide how to model the peak information within an xarray context. options are either to return a pandas dataframe, a data array, or label the input array.

    Parameters
    ----------

    x: Any
        A 1D ArrayLike signal containing peaks.
    find_peaks_kwargs: dict
        kwargs for `scipy.signal.find_peaks`
    peak_widths_kwargs: dict
        kwargs for `scipy.signal.peak_widths`
    """
    peak_dict = dict()

    peak_dict["p_idx"], props = signal.find_peaks(x, **find_peaks_kwargs)

    peak_dict["maxima"] = x[peak_dict["p_idx"]]
    (
        peak_dict["width"],
        peak_dict["width_height"],
        peak_dict["left_ip"],
        peak_dict["right_ip"],
    ) = signal.peak_widths(x=x, peaks=peak_dict["p_idx"], **peak_widths_kwargs)

    peak_table = pd.DataFrame(peak_dict).rename_axis("peak", axis=0)

    return peak_table


def plot_peaks(
    ds: xr.Dataset,
    group_dim: str = "",
    col_wrap: int = 1,
    input_signal_key: str = "",
    peak_table_key: str = "",
):
    """
    Draw the input signal overlaid with the peaks present in peak table

    TODO test
    TODO test without faceting
    TODO test with faceting.

    Parameters
    ----------

    peak_table: pd.DataFrame
        the peak table generated by `tabulate_peaks_2D`
    input_signal: Any, default = None
        the input 1D signal from which the peak table was generated. Supply to use as
        background to the peak mapping drawings.
    peak_outlines: bool, default = True
        draw the peak maxima and bases as an outline from the maxima to each base.
    peak_width_calc: bool, default = True
        draw the line calculating the peak widths as both the line drawn from maximum
        to the width height and then extending out to either peak boundary.
    """
    assert isinstance(ds, xr.Dataset)
    assert isinstance(group_dim, str)
    assert isinstance(col_wrap, int)
    assert col_wrap > 0
    assert isinstance(input_signal_key, str)
    assert isinstance(peak_table_key, str)

    subplots = _peak_traces(
        ds=ds,
        group_dim=group_dim,
        input_signal_key=input_signal_key,
        peak_table_key=peak_table_key,
    )

    # subplots.normalize_legend()

    # print(subplots.view_setting(["showlegend"]))

    fig = _draw_subplots_on_grid(subplots=subplots, col_wrap=col_wrap)

    titletext = f"'{input_signal_key}' peaks"

    if group_dim:
        titletext += f" faceted by '{group_dim}'"

    fig.update_layout(title_text=titletext)

    return fig


def _draw_subplots_on_grid(subplots: SubPlotCollection, col_wrap: int):
    grid = Grid(n_cells=len(subplots), col_wrap=col_wrap)
    fig = make_subplots(
        rows=grid.n_rows,
        cols=col_wrap,
        subplot_titles=subplots.titles,
    )

    for subplot, (idx, cell) in zip(subplots, grid.map_flat_grid().items()):
        fig.add_traces(subplot.traces, rows=cell.row, cols=cell.col)

    return fig


def _peak_traces(
    ds: xr.Dataset,
    group_dim: str,
    draw_peak_outlines: bool = True,
    draw_peak_width_calc: bool = True,
    input_signal_key: str = "",
    peak_table_key: str = "",
) -> SubPlotCollection:
    grpby = ds.groupby(group_dim)

    _subplots = []

    for key, sample in grpby:
        peak_table = get_peak_table_as_df(pt=sample.squeeze()[peak_table_key])

        input_signal = sample.squeeze()[input_signal_key].data

        subplot = SubPlot(
            peak_table=peak_table,
            input_signal_data=input_signal,
            sample=key,
            title=f"{group_dim} = {key}",
        )

        subplot.draw_traces()

        _subplots.append(subplot)

        subplots = SubPlotCollection(subplots=_subplots)

    return subplots


class PeakPicker:
    def __init__(self, da):
        """
        peak picker for XArray DataArrays.
        """

        if not isinstance(da, xr.DataArray):
            raise TypeError("expected DataArray")
        self._da = da
        self._peak_table = pd.DataFrame()
        self._dataarray = None
        self._pt_idx_cols = []

    def pick_peaks(
        self,
        core_dim: str | list = "",
        find_peaks_kwargs=dict(),
        peak_widths_kwargs=dict(),
    ) -> None:
        peak_tables = []

        # if x is not an iterable, make it so, so we can iterate over it. This is because x
        # can optionally be an iterable.

        da = self._da

        group_cols = [
            x
            for x in self._da.sizes.keys()
            if x not in (core_dim if isinstance(core_dim, list) else [core_dim])
        ]

        # after the above operation, if group_cols is an empty list, then the input
        # only contains one group.

        self._pt_idx_cols = [str(core_dim), "peak", "property"]

        if not group_cols:
            da = da.expand_dims("grp")
            group_cols = ["grp"]

        self._pt_idx_cols += group_cols

        # for each group in grouper get the group label and group
        for grp_label, group in da.groupby(group_cols):
            # generate the peak table for the current group
            peak_table = tablulate_peaks_1D(
                x=group.squeeze(),
                find_peaks_kwargs=find_peaks_kwargs,
                peak_widths_kwargs=peak_widths_kwargs,
            )

            # label each groups peak table with the group column name and values to provide
            # identifying columns for downstream joining etc.
            # works for multiple groups and single groups.
            for idx, val in enumerate(
                grp_label if isinstance(grp_label, tuple) else [grp_label]
            ):
                peak_table[group_cols[idx]] = val

            # add the core dim column subset by the peak indexes
            peak_table = peak_table.assign(
                **{core_dim: group[core_dim][peak_table["p_idx"].values].values}
            )

            peak_table = peak_table.reset_index()
            peak_tables.append(peak_table)

        peak_table = pd.concat(peak_tables)

        # remove helper group label
        if "grp" in self._pt_idx_cols:
            self._pt_idx_cols.remove("grp")
            self._peak_table = peak_table.drop("grp", axis=1)
        else:
            self._peak_table = peak_table

    @property
    def table(self):
        return self._peak_table

    @property
    def dataarray(self):
        """
        add the peak table as a dataarray. First time this is called it will generate
        the xarray DataArray attribute from the `table` attribute, storing the result
        internally for quicker returns if called again.
        """

        if self._dataarray is None:
            var_name = "property"
            id_vars = self._pt_idx_cols.copy()
            id_vars.remove(var_name)

            pt_da = (
                self._peak_table.melt(
                    id_vars=id_vars,
                    var_name=var_name,
                )
                .set_index(self._pt_idx_cols)
                .to_xarray()
                .to_dataarray(dim="value")
                .drop_vars("value")
                .squeeze()
            )
            pt_da.name = "peak_table"
            self._dataarray = pt_da
            return self._dataarray
        else:
            return self._dataarray


def compute_dataarray_peak_table(
    da: xr.DataArray, core_dim=None, find_peaks_kwargs=dict(), peak_widths_kwargs=dict()
) -> xr.DataArray:
    import pandas as pd

    if not isinstance(da, xr.DataArray):
        raise TypeError("expected DataArray")

    peak_tables = []

    # if x is not an iterable, make it so, so we can iterate over it. This is because x
    # can optionally be an iterable.

    group_cols = [
        x
        for x in da.sizes.keys()
        if x not in (core_dim if isinstance(core_dim, list) else [core_dim])
    ]

    # after the above operation, if group_cols is an empty list, then the input
    # only contains one group.

    pt_idx_cols = [str(core_dim), "peak", "property"]

    if not group_cols:
        da = da.expand_dims("grp")
        group_cols = ["grp"]

    pt_idx_cols += group_cols

    # for each group in grouper get the group label and group
    for grp_label, group in da.groupby(group_cols):
        # generate the peak table for the current group
        peak_table = tablulate_peaks_1D(
            x=group.squeeze(),
            find_peaks_kwargs=find_peaks_kwargs,
            peak_widths_kwargs=peak_widths_kwargs,
        )

        # label each groups peak table with the group column name and values to provide
        # identifying columns for downstream joining etc.
        # works for multiple groups and single groups.
        for idx, val in enumerate(
            grp_label if isinstance(grp_label, tuple) else [grp_label]
        ):
            peak_table[group_cols[idx]] = val

        # add the core dim column subset by the peak indexes
        peak_table = peak_table.assign(
            mins=group.mins[peak_table["p_idx"].values].values
        )

        peak_table = peak_table.reset_index()
        peak_tables.append(peak_table)

    peak_table = pd.concat(peak_tables)

    # remove helper group label
    if "grp" in pt_idx_cols:
        pt_idx_cols.remove("grp")
        peak_table = peak_table.drop("grp", axis=1)

    var_name = "property"
    id_vars = pt_idx_cols.copy()
    id_vars.remove(var_name)

    pt_da = (
        peak_table.melt(
            id_vars=id_vars,
            var_name=var_name,
        )
        .set_index(pt_idx_cols)
        .to_xarray()
        .to_dataarray(dim="value")
        .drop_vars("value")
        .squeeze()
    )
    pt_da.name = "peak_table"

    return pt_da
