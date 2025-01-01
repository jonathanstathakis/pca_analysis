"""
In chromatography, a peak table generally consists of the time location of the maxima, the left and right bounds, the width and the area.

`scipy.signal` provides functions for obtaining the location, height, width, prominence, left and right base of detected peaks. Other parameters can be derived from these fundamental ones.
"""

import xarray as xr
from . import xr_signal
import plotly.graph_objects as go
from itertools import cycle

PEAKS = "peaks"


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


from scipy import signal
import pandas as pd


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
    peak_table: pd.DataFrame,
    input_signal=None,
    peak_outlines: bool = True,
    peak_width_calc: bool = True,
):
    """
    Draw the input signal overlaid with the peaks present in peak table

    TODO test

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
    import plotly.io as pio

    colors = cycle(
        pio.templates[pio.templates.default]["layout"]["colorscale"]["diverging"]
    )

    fig = go.Figure()

    if input_signal is not None:
        # input signal

        fig.add_trace(go.Scatter(y=input_signal, name="signal"))

    for color, (idx, row) in zip(colors, peak_table.iterrows()):
        if peak_outlines:
            # plot the peaks as mapped by `peak_widths`
            # the base height is `width_height`

            fig.add_trace(
                go.Scatter(
                    x=[row["left_ip"], row["p_idx"], row["right_ip"]],
                    y=[row["width_height"], row["maxima"], row["width_height"]],
                    name="outline",
                    mode="lines",
                    line=dict(color=color[1], width=1),
                    legendgroup=idx,
                    legendgrouptitle_text=idx,
                )
            )

        if peak_width_calc:
            fig.add_trace(
                go.Scatter(
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
                    line=dict(dash="dot", width=0.75, color=color[1]),
                    legendgroup=idx,
                    legendgrouptitle_text=idx,
                ),
            )

    fig = fig.update_layout(title=dict(text="peak mapping"))
    return fig
