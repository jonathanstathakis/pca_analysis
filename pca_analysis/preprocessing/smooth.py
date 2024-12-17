from scipy.signal import savgol_filter
import xarray as xr


def savgol_smooth(da: xr.DataArray, var_key: str, savgol_kwargs):
    # smoothing
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
    da_ = xr.apply_ufunc(
        savgol_filter,
        da[var_key],
        kwargs={**dict(axis=1), **savgol_kwargs},
    )

    return da_
