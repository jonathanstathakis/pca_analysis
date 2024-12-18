"""
Functions applying signal processing techniques to xarray data structures.

Many default labels can be overriden by accessing the corresponding constant object

Usage
-----

Use `find_peaks_dataset` as a Dataset level API for peak finding and results viz.
Otherwise `find_peaks_array` can be used to find the peaks in a xr.DataArray, and
`facet_plot_signal_peaks` can be used to overlay the peaks on the input signal if
the input signal and peaks are combined into a Dataset.

TODO4 describe constants in docs.
TODO4 test whether the overriding works.
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

NEW_ARR_NAME = "peaks"
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
    TODO4 document me
    TODO4 check that the props are being handled as expected. Expect a nested mapping
    for each group.
    """

    peaks = []

    for k, v in da.groupby(group=grouper):
        signal = v.squeeze()
        peak_idxs, props_ = sp_find_peaks(signal, **find_peaks_kws)

        peak_xr = v.isel(**{f"{x_key}": peak_idxs})
        peak_xr.assign_attrs({f"{PROPS_PREFIX}_{k}": props_})
        peak_xr.name = NEW_ARR_NAME
        peaks.append(peak_xr)

    peaks_ = xr.merge(peaks).assign_attrs(
        dict(description=PEAKS_ARRAY_DESC, parameters=find_peaks_kws)
    )

    return peaks_


def facet_plot_multiple_traces(
    ds: xr.Dataset,
    grouper=None,
    var_keys=None,
    x_key=None,
    col_wrap: int = 1,
    fig_kwargs={},
    trace_kwargs: dict = {},
) -> go.Figure:
    """Facet plotting of xarray DataArrays in a dataset. Developed for overlaying
    chromatographic peak maxima over the input signal.

    Iterates over the dataset on `grouper`, accessing the DataArray data of each group
    to produce the plots.

    TESTME

    Parameters
    ----------

    ds : xr.Dataset
        a dataset containing two applicable `xr.DataArray`
    grouper  : list[str]
        a list of strings representing distinct groups within the dataset.
    var_keys : list[str]
        the labels of each data array to plot. The corresponding trace style
        args are passed through `trace_kwargs` in the same order. For example if you
        want to plot a facet of a scatter overlaying a line, set trace 1 to 'markers'
        and trace 2 to 'line' (?) TODO verify.
    x : str
        the coordinate label to be used for the x axis.
    y : str
        the label to be used for the y axis (only used for labeling the axis, not
        accessing)
    col_wrap: int
        optional maximum number of columns in subplot grid before beginning a new row.
    fig_kwargs: dict
        optional kwargs to be passed to `plotly.subplots.make_subplots`
    trace_kwargs: dict
        optional kwargs to be passed to `plotly.go.Scatter' for the trace. keys must be the same as in `var_keys`.
    """

    # trace_kwargs is kind of compulsory but to soften the learning curve set default
    # mode to markers.
    # if var_keys isnt provided, plot them all

    if not var_keys:
        var_keys = list(ds.keys())

    groups = ds.groupby(grouper)
    n_plots = len(groups)
    n_rows = int(np.ceil(n_plots / col_wrap))

    fig_kwargs["x_title"] = x_key

    fig = make_subplots(
        rows=n_rows,
        cols=col_wrap,
        subplot_titles=list(str(x) for x in dict(groups).keys()),
        **fig_kwargs,
    )

    curr_col = 1
    curr_row = 1

    # unpacking the dict consumes it. need to copy
    trace_kwargs_ = trace_kwargs.copy()

    import plotly.express as px

    colormap = px.colors.qualitative.Plotly[: len(var_keys)]

    for grp_idx, (grp_key, grp) in enumerate(groups):
        for var_key, color in zip(var_keys, colormap):
            # signal
            y = grp[var_key].squeeze()

            x = grp[var_key].coords.get(x_key)

            if x:
                if x.ndim != 1:
                    raise ValueError(f"{x.ndim=}. input ds as follows: {ds.sizes}")
            if y.ndim != 1:
                raise ValueError(f"{y.ndim=}. input ds as follows: {ds.sizes}")
            trace_name = grp[var_key].name

            if grp_idx == 0:
                showlegend = True
            else:
                showlegend = False

            # trace kwargs are optional but need to unpack a dict
            # to satisfy the go.Scatter call below
            if trace_kwargs.get(var_key):
                trace_kwargs_ = trace_kwargs[var_key]
            else:
                trace_kwargs_ = {}

            trace = go.Scatter(
                x=x,
                y=y,
                name=trace_name,
                showlegend=showlegend,
                legendgroup=str(var_key),
                marker=dict(color=color),
                **trace_kwargs_,
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
    new_arr_name: str = NEW_ARR_NAME,
    return_viz: bool = False,
) -> xr.Dataset:
    """apply find peaks to a DataArray of `ds`, assign the resulting peaks DataArray
    back and optionally provide a viz of the peaks overlaying the signals faceted by
    `grouper`.

    if `return_viz` is True, the function will return a faceted
    plot on distinct `grouper` alongside the modified Dataset,
    otherwise it will only return the Dataset.

    This is a convenience function to combine the calculation and viz in one call.

    Parameters
    ----------
    TODO4 describe parameters
    TESTME
    """

    peaks = find_peaks_array(
        da=ds[array_key], grouper=grouper, find_peaks_kws=find_peaks_kws, x_key=x_key
    )

    if new_arr_name != NEW_ARR_NAME:
        peaks = peaks.rename({NEW_ARR_NAME: new_arr_name})
    ds = ds.merge(peaks)

    if return_viz:
        raise RuntimeError(
            "return_viz is deprecated, instead call facet_plot_multiple_traces directly"
        )

    return ds
