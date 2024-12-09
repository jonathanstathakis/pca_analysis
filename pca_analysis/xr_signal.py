"""
Functions applying signal processing techniques to xarray data structures.

Many default labels can be overriden by accessing the corresponding constant object

Usage
-----

Use `find_peaks_dataset` as a Dataset level API for peak finding and results viz.
Otherwise `find_peaks_array` can be used to find the peaks in a xr.DataArray, and
`facet_plot_signal_peaks` can be used to overlay the peaks on the input signal if
the input signal and peaks are combined into a Dataset.

TODO describe constants in docs.
TODO test whether the overriding works.
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
PROPS_PREFIX = "props"
PEAKS_ARRAY_DESC = "detected peaks"


def find_peaks_array(
    da: xr.DataArray,
    grouper: list[str],
    x_key: str,
    find_peaks_kws: dict = find_peaks_defaults,
) -> xr.Dataset:
    """
    at the moment doesnt attempt to store the peak properties as the scipy logic
    is annoying and the result does not align with the input data or peak data withhout
    some decisions being made about where a peak is.

    Cells without peaks are left nan.

    TESTME
    TODO document me
    TODO check that the props are being handled as expected. Expect a nested mapping
    for each group.
    """

    peaks = []

    for k, v in da.groupby(group=grouper):
        signal = v.squeeze()
        peak_idxs, props_ = sp_find_peaks(signal, **find_peaks_kws)

        peak_xr = v.isel(**{f"{x_key}": peak_idxs})
        peak_xr.assign_attrs({f"{PROPS_PREFIX}_{k}": props_})
        peak_xr.name = PEAKS_XARRAY_NAME
        peaks.append(peak_xr)

    peaks_ = xr.merge(peaks).assign_attrs(
        dict(description=PEAKS_ARRAY_DESC, parameters=find_peaks_kws)
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


def find_peaks_dataset(
    ds: xr.Dataset,
    array_key: str,
    grouper: list[str],
    x_key: str,
    find_peaks_kws: dict = find_peaks_defaults,
    return_viz: bool = True,
    y: str = "",
    col_wrap: int = 1,
    fig_kwargs={},
    lines_kwargs={},
    peaks_kwargs={},
) -> xr.Dataset | tuple[xr.Dataset, go.Figure]:
    """apply find peaks to a DataArray of `ds`, assign the resulting peaks DataArray
    back and optionally provide a viz of the peaks overlaying the signals faceted by
    `grouper`.

    if `return_viz` is True, the function will return a faceted
    plot on distinct `grouper` alongside the modified Dataset,
    otherwise it will only return the Dataset.

    This is a convenience function to combine the calculation
    and viz in one call.

    Parameters
    ----------
    TODO describe parameters
    TESTME
    """

    peaks = find_peaks_array(
        da=ds[array_key], grouper=grouper, find_peaks_kws=find_peaks_kws, x_key=x_key
    )

    ds = ds.merge(peaks)

    if return_viz:
        fig = facet_plot_signal_peaks(
            ds=ds,
            grouper=grouper,
            line_key=array_key,
            marker_key=PEAKS_XARRAY_NAME,
            x=x_key,
            y=y,
            col_wrap=col_wrap,
            fig_kwargs=fig_kwargs,
            lines_kwargs=lines_kwargs,
            peaks_kwargs=peaks_kwargs,
        )

        return ds, fig
    else:
        return ds
