import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pca_analysis.peak_picking import (
    get_peak_table_as_df,
)


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


def draw_signal_trace(signal_data, sample, color):
    signal_trace = go.Scatter(
        y=signal_data,
        name="signal",
        showlegend=False,
        meta=dict(sample=str(sample)),
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
                draw_maxima_traces(color=color, row=row, idx=int(idx), sample=str(key))
            )

            width_calc_traces.append(
                draw_peak_width_calc_traces(
                    color=color, row=row, idx=int(idx), sample=str(key)
                )
            )
            grp_traces += outline_traces
            grp_traces += maxima_traces
            grp_traces += width_calc_traces
        traces.append(grp_traces)
    return traces


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

    sample_0 = fig.data[0]["meta"]["sample"]
    # show legend for first peak trace markers
    fig.for_each_trace(
        lambda trace: trace.update(showlegend=True)
        if trace.meta["sample"] == sample_0 and trace.meta["peak"] == 0
        else ()
        if "peak" in trace.data["meta"]
        else ()
    )

    # show legend for first signal
    fig.for_each_trace(
        lambda trace: trace.update(showlegend=True)
        if trace.name == "signal" and trace.meta["sample"] == sample_0
        else ()
    )

    return fig
