"""
Functions applying signal processing techniques to xarray data structures
"""

from scipy.signal import find_peaks as sp_find_peaks
import pandas as pd
import xarray as xr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

find_peaks_defaults = dict(
    height=None,
    threshold=None,
    distance=None,
    prominence=None,
    width=None,
    wlen=None,
    rel_height=0.5,
    plateau_size=None,
)

PEAKS_XARRAY_NAME = "peaks"


def find_peaks(
    da: xr.DataArray, dim_order: list[str], find_peaks_kws: dict = find_peaks_defaults
):
    """
    at the moment doesnt attempt to store the peak properties as the scipy logic
    is annoying and the result does not align with the input data or peak data withhout
    some decisions being made about where a peak is.

    TESTME
    TODO document me
    """

    peaks = []
    for k, v in da.to_dataframe(
        name=PEAKS_XARRAY_NAME,
        dim_order=dim_order,
    ).groupby(as_index=False, group_keys=False, by=["sample", "mz"]):
        signal = v.to_numpy().ravel()
        peak_idxs, props_ = sp_find_peaks(signal, **find_peaks_kws)
        peaks.append(v.iloc[peak_idxs])

    peaks_ = (
        pd.concat(peaks)
        .to_xarray()
        .assign_attrs(
            dict(description="detected peaks", parameters=find_peaks_kws, props=props_)
        )
    )

    return peaks_


def facet_plot_signal_peaks(
    ds: xr.Dataset,
    grouper: list[str],
    line_key: str,
    marker_key: str,
    x: str,
    y: str,
    col_wrap: int = 1,
    fig_kwargs={},
    lines_kwargs={},
    peaks_kwargs={},
):
    """Facet plotting of two xarray Datasets, where one is best represented as lines and
    the other as markers. Developed for overlaying chromatographic peak maxima over the
    input signal.

    Iterates over the dataset on `grouper`, accessing the DataArray data of each group
    to produce the plots.

    TESTME

    Parameters
    ----------

    ds : xr.Dataset
         a dataset containing two applicable `xr.DataArray`
    grouper  : list[str]
               a list of strings representing distinct groups within the dataset.
    line_key : str
               the access key of the `xr.DataArray` pertaining to the line data.
    marker_key : str
                the access key of the `xr.DataArray` pertaining to the marker data.
    x : str
        the coordinate label to be used for the x axis.
    y : str
        the label to be used for the y axis (only used for labeling the axis, not
        accessing)
    fig_kwargs: dict
        optional kwargs to be passed to `plotly.subplots.make_subplots`
    lines_kwargs: dict
        optional kwargs to be passed to the `go.Scatter` init for the line traces.
    peaks_kwargs: dict
        optional kwargs to be passed to the `go.Scatter` init for the marker traces.
    col_wrap: int
        optional maximum number of columns in subplot grid before beginning a new row.
    """
    groups = ds.groupby(grouper)
    n_plots = len(groups)
    n_rows = int(np.ceil(n_plots / col_wrap))

    fig = make_subplots(
        rows=n_rows,
        cols=col_wrap,
        subplot_titles=list(str(x) for x in dict(groups).keys()),
        x_title=x,
        y_title=y,
        **fig_kwargs,
    )

    line_color = dict(color="blue")
    peak_color = dict(color="red")
    legendgroup_lines = "line"
    legendgroup_peaks = "peaks"

    curr_col = 1
    curr_row = 1

    for idx, (k, v) in enumerate(groups):
        # signal
        line_y = v[line_key].squeeze()
        line_x = v[line_key].coords[x]
        line_name = v[line_key].name

        if idx == 0:
            showlegend = True
        else:
            showlegend = False
        trace = go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            name=line_name,
            marker=line_color,
            showlegend=showlegend,
            legendgroup=legendgroup_lines,
            **lines_kwargs,
        )

        fig.add_trace(trace, row=curr_row, col=curr_col)

        # peaks
        peak_y = v[marker_key].squeeze()
        peak_x = line_x
        peaks_name = v[marker_key].name

        trace = go.Scatter(
            x=peak_x,
            y=peak_y,
            name=peaks_name,
            mode="markers",
            marker=peak_color,
            showlegend=showlegend,
            legendgroup=legendgroup_peaks,
            **peaks_kwargs,
        )

        fig.add_trace(trace, row=curr_row, col=curr_col)

        # reset cols and move to next row if max col reached.
        if curr_col == col_wrap:
            curr_col = 1
            curr_row += 1
        else:
            curr_col += 1

    return fig
