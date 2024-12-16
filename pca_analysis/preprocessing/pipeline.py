# TODO fine tune clustering to goup the first 5 peaks together
# TODO write tests for 'clustering_by_maxima' module.

"""
Notes
=====

Use xr.DataArray as first class object for majority of operations. Use the dataset
as a global store.

library consists of the following:

1. baseline correction
2. smoothing/sharpening
3. peak picking / grouping
4. deconvolution
"""

from sklearn import cluster
import plotly.graph_objects as go
import pandas as pd
from ipywidgets import widgets
from scipy.signal import savgol_filter
import xarray as xr
from .. import xr_signal
import plotly.express as px


PEAKS = "peaks"


def smooth(ds, var_key: str, savgol_kwargs):
    # smoothing
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
    ds = ds.assign(
        data_corr_smooth=xr.apply_ufunc(
            savgol_filter,
            ds[var_key],
            kwargs={**dict(axis=1), **savgol_kwargs},
        )
    )

    return ds


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


def clustering_by_maxima(
    ds: xr.Dataset,
    signal_key: str,
    x_key: str,
    grouper: list[str],
    facet_peak_plot_kwargs: dict = {},
    savgol_kwargs: dict = {},
    by_maxima: bool = False,
    display_facet_peaks_plot: bool = True,
    display_cluster_table: bool = True,
    find_peaks_kws=dict(
        rel_height=0.5,
        prominence=3,
        height=-5,
        distance=150,
    ),
    clustering_kws: dict = dict(n_clusters=None, linkage="average"),
):
    """
    if by_maxima = True, cluster on the maxima, otherwise cluster the minima


    Notes
    =====

    Setting `n_clusters` to None (default) appears to assign labels to the whole input array.

    """

    if savgol_kwargs:
        ds = ds.pipe(smooth, signal_key, savgol_kwargs)

    ds = find_peaks(
        ds=ds,
        signal_key=signal_key,
        by_maxima=by_maxima,
        find_peaks_kws=find_peaks_kws,
        grouper=grouper,
        x_key=x_key,
        maxima_coord_name=PEAKS,
    )

    figures = []

    if display_facet_peaks_plot:
        # add common and scope specific kwargs
        facet_peak_plot_kwargs["data_keys"] = [signal_key, PEAKS]
        facet_peak_plot_kwargs["grouper"] = grouper
        facet_peak_plot_kwargs["x"] = x_key
        facet_peak_plot_kwargs["trace_kwargs"] = [
            dict(mode="lines", marker=dict(size=3)),
            dict(mode="markers", marker=dict(size=3, opacity=0.8)),
        ]

        # generate a peak plot for each sample
        facet_peak_fig = ds.pipe(
            xr_signal.facet_plot_multiple_traces, **facet_peak_plot_kwargs
        )

        # figures to displayed in the final widget.
        figures.append(facet_peak_fig)

    # extract the computed minima
    # we dont care about sample labeling as we're treating all sample peaks as
    # part of the same set.

    ds = ds.pipe(assign_cluster_labels, clustering_kws)

    # the left and right bounds of each cluster, used below to draw them
    cluster_bounds = ds.coords["cluster"].pipe(get_cluster_bounds)

    # viz the clusters overlaying the signal
    cluster_regions_fig = cluster_signal_peak_overlay(
        ds=ds,
        signal_key=signal_key,
        by_maxima=by_maxima,
        time_ordered_cluster_bounds=cluster_bounds,
    )

    figures.append(cluster_regions_fig)

    display(cluster_bounds)

    # organise results figures into an ipywidget
    title = widgets.HTML("<h1>Clustering By Minima</h1>")
    figure_widgets = [go.FigureWidget(x) for x in figures]

    if display_cluster_table:
        figure_widgets.append(widgets.HTML(cluster_bounds.to_html()))

    box = widgets.VBox(
        [title] + figure_widgets,
    )

    return ds, box


def get_cluster_bounds(da: xr.DataArray):
    time_ordered_cluster_bounds = (
        da.to_pandas()
        .to_frame()
        .reset_index()
        .reset_index()
        .groupby("cluster")["mins"]
        .agg(left_bound="min", right_bound="max")
    )

    return time_ordered_cluster_bounds


def cluster_signal_peak_overlay(ds, signal_key, by_maxima, time_ordered_cluster_bounds):
    cluster_regions_fig = go.Figure()

    # add signal trace
    for idx, grp in (
        ds[signal_key]
        .isel(id_rank=slice(None, None, 3))
        .to_dataframe()
        .reset_index()
        .groupby("id_rank")
    ):
        cluster_regions_fig.add_trace(
            go.Scatter(x=grp["mins"], y=grp[signal_key], line=dict(width=1)),
        )

    # add cluster rectangles

    for (idx, clus), color in zip(
        time_ordered_cluster_bounds.iterrows(), px.colors.qualitative.Plotly
    ):
        cluster_regions_fig.add_vrect(
            x0=clus["left_bound"],
            x1=clus["right_bound"],
            fillcolor=color,
            opacity=0.2,
            label=dict(text=idx, textposition="top center"),
        )

    # add the clustered samplewise minima
    samplewise_peak_minima = ds.peaks.to_dataframe().reset_index()
    minima_x = samplewise_peak_minima["mins"]
    minima_y = samplewise_peak_minima["peaks"]

    cluster_regions_fig.add_trace(
        go.Scatter(
            x=minima_x,
            y=minima_y if by_maxima else minima_y * -1,
            mode="markers",
            marker=dict(size=4, color="red", opacity=0.8),
        )
    )

    return cluster_regions_fig


def assign_cluster_labels(ds, clustering_kws):
    maxima = ds.peaks.to_dataframe().reset_index()[["id_rank", "mins", "peaks"]]
    cluster_input = maxima.dropna().sort_values(["mins"])
    # label the clusters

    agg_clus = cluster.AgglomerativeClustering(**clustering_kws)

    cluster_labels = agg_clus.fit_predict(cluster_input[["mins"]])
    cluster_input["cluster"] = cluster_labels

    maxima = maxima.join(cluster_input["cluster"])

    # fill to label the clusters across the whole dataset. intercluster regions
    # are labelled '-1'.
    maxima = (
        maxima.assign(cluster=lambda x: x["cluster"].ffill().fillna(-1).astype(int))
        .drop_duplicates("mins")
        .sort_values(["mins"])
    )

    # assign cluster labels to ds
    ds = ds.assign_coords(cluster=("mins", maxima["cluster"]))

    # compute the left and right bounds of each cluster
    cluster_bounds = (
        ds.to_dataframe()
        .reset_index()
        .groupby("cluster")["mins"]
        .agg(left_bound="min", right_bound="max")
        .reset_index()
    )

    # drop interpeak region as we dont want to label it currently.
    ordered_bounds = (
        cluster_bounds[["cluster", "left_bound"]]
        .loc[lambda x: x["cluster"] != -1]
        .assign(
            left_rank=lambda x: x["left_bound"].rank(method="dense").sub(1).astype(int)
        )[["cluster", "left_rank"]]
    )

    # the cluster labels positionally ordered corresponding to the mins
    # coordinate of ds
    time_ordered_cluster_labels = (
        maxima[["cluster", "mins"]]
        .pipe(
            pd.merge,
            right=ordered_bounds,
            on="cluster",
            how="left",
        )
        .drop("cluster", axis=1)
        .assign(cluster=lambda x: x["left_rank"].fillna(-1).astype(int))
        .drop(["left_rank"], axis=1)
    )

    # add the time ordered labels to the dataset

    ds = ds.assign_coords(cluster=("mins", time_ordered_cluster_labels["cluster"]))

    ds = ds.assign_attrs(
        {
            "notes": "'cluster' coordinate is labelled in time ascending order, except -1 which is used for unclustered portions of the time axis"
        }
    )

    return ds
