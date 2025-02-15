"""
In chromatography, a peak table generally consists of the time location of the maxima, the left and right bounds, the width and the area.

`scipy.signal` provides functions for obtaining the location, height, width, prominence, left and right base of detected peaks. Other parameters can be derived from these fundamental ones.
"""

import xarray as xr
from . import xr_signal
from scipy import signal
import pandas as pd
from dataclasses import dataclass


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
    x,
    find_peaks_kwargs=dict(),
    peak_widths_kwargs=dict(rel_height=0.95),
    timestep: float = -1,
) -> pd.DataFrame:
    """
    Generate a peak table for a 1D input signal x.

    TODO4 test

    Parameters
    ----------

    x: Any
        A 1D ArrayLike signal containing peaks.
    find_peaks_kwargs: dict
        kwargs for `scipy.signal.find_peaks`
    peak_widths_kwargs: dict
        kwargs for `scipy.signal.peak_widths`

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame representing the peak table with columns:
        "p_idx", "maxima", "width", "width_height", "left_ip", and
        "right_ip", and index "peak". "peak" is the numeric order of the
        peaks, starting at 0, and also the row index. "p_idx" is the indice
        of the peak maxima in the input `x`, "maxima" is the magnitude of the
        signal at "p_idx", "width" is the interpolatd length of the peak in in index units, "width_height" is the interpolated magnitude of the signal at the point at which the width is measured, "left_ip" and "right_ip" is the interpolated intersection point of the width measurement with the signal.


    """
    peak_dict = dict()

    peak_dict["p_idx"], props = signal.find_peaks(x, **find_peaks_kwargs)

    peak_dict["maxima"] = x[peak_dict["p_idx"]]
    (
        peak_dict["width"],
        peak_dict["width_height"],
        peak_dict["left_ip"],
        peak_dict["right_ip"],
    ) = signal.peak_widths(
        x=x,
        peaks=peak_dict["p_idx"],
        **peak_widths_kwargs,
    )

    peak_table = pd.DataFrame(peak_dict).rename_axis("peak", axis=0)

    # add a timestep x column to provide the values in time units
    peak_table = peak_table.assign(**{"maxima_x": lambda x: x["p_idx"]})

    # convert idx units to mins
    for key in ["width", "left_ip", "right_ip", "maxima_x"]:
        peak_table = peak_table.assign(**{key: lambda a: a[key] * timestep})

    return peak_table


class PeakPicker:
    def __init__(
        self,
        da,
    ):
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
        peak_widths_kwargs=dict(rel_height=1),
        x_key: str = "",
    ) -> None:
        peak_tables = []

        # if x is not an iterable, make it so, so we can iterate over it. This is because x can optionally be an iterable.

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

        if x_key:
            timestep = da[x_key].diff(x_key).mean().item()
        else:
            timestep = -1

        # for each group in grouper get the group label and group
        for grp_label, group in da.groupby(group_cols):
            # generate the peak table for the current group
            peak_table = tablulate_peaks_1D(
                x=group.squeeze(),
                timestep=timestep,
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
