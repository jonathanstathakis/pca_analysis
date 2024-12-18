from scipy.signal import savgol_filter
import xarray as xr


def savgol_smooth(
    da: xr.DataArray, input_core_dims=None, output_core_dims=((),), **savgol_kwargs
):
    # smoothing
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
    da_ = xr.apply_ufunc(
        savgol_filter,
        da,
        input_core_dims=input_core_dims,
        output_core_dims=output_core_dims,
        kwargs=savgol_kwargs,
        vectorize=True,
        # kwargs={**dict(axis=1), **savgol_kwargs},
    )

    return da_
