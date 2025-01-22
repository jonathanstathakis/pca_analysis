import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pca_analysis.peak_picking import (
    get_peak_table_as_df,
)
from typing import Any
from pandas import Series

import xarray as xr


class Cell:
    def __init__(self, row, col):
        self.row = row
        self.col = col


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


class PeakPlotTracer:
    def __init__(self, group_dim, group_key):
        self.group_dim = str(group_dim)
        self.group_key = str(group_key)

    def _gen_meta(self, peak_idx):
        return {self.group_dim: self.group_key, "peak": peak_idx}

    def draw_signal_trace(self, signal_x, signal_y, color):
        signal_trace = go.Scatter(
            x=signal_x,
            y=signal_y,
            name="signal",
            showlegend=False,
            meta=self._gen_meta(peak_idx=None),
            legendgroup="signal",
            line_color=color,
        )

        return signal_trace

    def draw_outline_traces(
        self,
        peak_idx: int,
        row: Series,
        color,
    ):
        outline_trace = go.Scatter(
            x=[row["left_ip"], row["maxima_x"], row["right_ip"]],
            y=[row["width_height"], row["maxima"], row["width_height"]],
            name="outline",
            mode="lines",
            line_color=color,
            line_width=1,
            legendgroup="outline",
            showlegend=False,
            meta=self._gen_meta(peak_idx=peak_idx),
        )

        return outline_trace

    def draw_peak_width_calc_traces(self, row: Series, color, peak_idx: int):
        trace = go.Scatter(
            x=[
                row["left_ip"],
                row["right_ip"],
                None,
                row["maxima_x"],
                row["maxima_x"],
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
            meta=self._gen_meta(peak_idx=peak_idx),
            customdata=[],
        )

        return trace

    def draw_maxima_traces(self, row: Series, color, peak_idx: int):
        maxima_trace = go.Scatter(
            x=[row["maxima_x"]],
            y=[row["maxima"]],
            mode="markers",
            name="maxima",
            marker_color=color,
            marker_size=5,
            legendgroup="maxima",
            showlegend=False,
            meta=self._gen_meta(peak_idx=peak_idx),
        )

        return maxima_trace


def _peak_traces(
    ds: xr.Dataset,
    x: str,
    group_dim: str,
    draw_peak_outlines: bool = True,
    draw_peak_width_calc: bool = True,
    input_signal_key: str = "",
    peak_table_key: str = "",
):
    grpby = ds.groupby(group_dim)

    colormap = px.colors.qualitative.Plotly

    signal_color = colormap[0]
    traces: dict[Any, list] = {}
    for grp_key, sample in grpby:
        traces_key = f"{group_dim}='{grp_key}'"
        traces[traces_key] = []

        signal_x = sample.squeeze()[x]
        signal_y = sample.squeeze()[input_signal_key].data

        ppt = PeakPlotTracer(group_dim=group_dim, group_key=grp_key)

        traces[traces_key] += [
            ppt.draw_signal_trace(
                signal_x=signal_x,
                signal_y=signal_y,
                color=signal_color,
            )
        ]

        peak_colors = colormap[1:]
        peak_table = get_peak_table_as_df(pt=sample.squeeze()[peak_table_key])

        # each iteration draws the traces for one peak.
        outline_traces = []
        maxima_traces = []
        width_calc_traces = []
        for color, (peak_idx, row) in zip(peak_colors, peak_table.iterrows()):
            if not isinstance(peak_idx, int):
                raise TypeError
            outline_traces.append(
                ppt.draw_outline_traces(
                    peak_idx=peak_idx,
                    row=row,
                    color=color,
                )
            )
            maxima_traces.append(
                ppt.draw_maxima_traces(
                    color=color,
                    row=row,
                    peak_idx=peak_idx,
                )
            )

            width_calc_traces.append(
                ppt.draw_peak_width_calc_traces(
                    color=color,
                    row=row,
                    peak_idx=peak_idx,
                )
            )
        # adds the traces for all the peaks for the group. Information
        # about what group each trace belongs to is stored internally
        # in name and in metadata.
        traces[traces_key] += outline_traces
        traces[traces_key] += maxima_traces
        traces[traces_key] += width_calc_traces
    return traces


def plot_peaks(
    ds: xr.Dataset,
    x: str = "",
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
    assert isinstance(x, str)

    subplots = _peak_traces(
        ds=ds,
        x=x,
        group_dim=group_dim,
        input_signal_key=input_signal_key,
        peak_table_key=peak_table_key,
    )

    grid = Grid(n_cells=len(subplots), col_wrap=col_wrap)
    fig = make_subplots(
        rows=grid.n_rows,
        cols=col_wrap,
        subplot_titles=[str(x) for x in subplots.keys()],
        x_title=x,
        y_title=input_signal_key,
    )

    for subplot_traces, (idx, cell) in zip(
        list(subplots.values()), grid.map_flat_grid().items()
    ):
        fig.add_traces(subplot_traces, rows=cell.row, cols=cell.col)
        assert True

    titletext = f"'{input_signal_key}' peaks"

    if group_dim:
        titletext += f" faceted by '{group_dim}'"

    fig.update_layout(title_text=titletext)

    sample_0 = fig.data[0]["meta"]["sample"]

    # show legend for first peak trace markers
    fig.for_each_trace(
        lambda trace: trace.update(showlegend=True)
        if "peak" in trace["meta"]
        and trace.meta["sample"] == sample_0
        and trace.meta["peak"] == 0
        else ()
    )

    # show legend for first signal
    fig.for_each_trace(
        lambda trace: trace.update(showlegend=True)
        if trace.name == "signal" and trace.meta["sample"] == sample_0
        else trace
    )

    return fig
