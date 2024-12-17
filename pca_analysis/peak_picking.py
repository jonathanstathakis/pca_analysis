import xarray as xr
from . import xr_signal

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
