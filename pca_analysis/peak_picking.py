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

        assert isinstance(n_cells, int)
        assert isinstance(col_wrap, int)

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


def draw_signal_trace(signal_data, sample, color):
    signal_trace = go.Scatter(
        y=signal_data,
        name="signal",
        showlegend=False,
        meta=dict(sample=sample),
        legendgroup="signal",
        line_color=color,
    )

    return signal_trace


def draw_outline_traces(
    idx,
    row,
    color,
    sample,
):
    outline_trace = go.Scatter(
        x=[row["left_ip"], row["p_idx"], row["right_ip"]],
        y=[row["width_height"], row["maxima"], row["width_height"]],
        name="outline",
        mode="lines",
        line_color=color,
        line_width=1,
        legendgroup="outline",
        showlegend=False,
        meta=dict(peak=int(idx), sample=str(sample)),
    )

    return outline_trace


# for color, (idx, row) in zip(self.peak_colors, self.peak_table.iterrows())


def draw_peak_width_calc_traces(row, color, idx, sample):
    trace = go.Scatter(
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
        meta=dict(peak=int(idx), sample=str(sample)),
        customdata=[],
    )

    return trace


def draw_maxima_traces(row, color, idx, sample):
    maxima_trace = go.Scatter(
        x=[row["p_idx"]],
        y=[row["maxima"]],
        mode="markers",
        name="maxima",
        marker_color=color,
        marker_size=5,
        legendgroup="maxima",
        showlegend=False,
        meta=dict(peak=int(idx), sample=str(sample)),
    )

    return maxima_trace


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

    fig = _draw_subplots_on_grid(subplots=subplots, col_wrap=col_wrap)

    titletext = f"'{input_signal_key}' peaks"

    if group_dim:
        titletext += f" faceted by '{group_dim}'"

    fig.update_layout(title_text=titletext)

    return fig


def _draw_subplots_on_grid(subplots, col_wrap: int):
    grid = Grid(n_cells=len(subplots), col_wrap=col_wrap)
    fig = make_subplots(
        rows=grid.n_rows,
        cols=col_wrap,
        # subplot_titles=subplots.titles,
    )

    for subplot_traces, (idx, cell) in zip(subplots, grid.map_flat_grid().items()):
        fig.add_traces(subplot_traces, rows=cell.row, cols=cell.col)
        assert True

    return fig


def _peak_traces(
    ds: xr.Dataset,
    group_dim: str,
    draw_peak_outlines: bool = True,
    draw_peak_width_calc: bool = True,
    input_signal_key: str = "",
    peak_table_key: str = "",
):
    grpby = ds.groupby(group_dim)

    colormap = px.colors.qualitative.Plotly

    signal_color = colormap[0]
    traces = []
    for key, sample in grpby:
        grp_traces = []
        peak_table = get_peak_table_as_df(pt=sample.squeeze()[peak_table_key])

        input_signal = sample.squeeze()[input_signal_key].data

        peak_colors = colormap[1:]

        grp_traces.append(
            draw_signal_trace(
                signal_data=input_signal,
                sample=key,
                color=signal_color,
            )
        )
        outline_traces = []
        maxima_traces = []
        width_calc_traces = []
        for color, (idx, row) in zip(peak_colors, peak_table.iterrows()):
            outline_traces.append(
                draw_outline_traces(
                    idx=idx,
                    row=row,
                    color=color,
                    sample=key,
                )
            )
            maxima_traces.append(
                draw_maxima_traces(color=color, row=row, idx=idx, sample=sample)
            )

            width_calc_traces.append(
                draw_peak_width_calc_traces(
                    color=color, row=row, idx=idx, sample=sample
                )
            )
            grp_traces += outline_traces
            grp_traces += maxima_traces
            grp_traces += width_calc_traces
        traces.append(grp_traces)
    return traces


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
