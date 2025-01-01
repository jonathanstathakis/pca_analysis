"""
Provide methods of sharpening signals
"""

from scipy import ndimage
import xarray as xr


def laplacian(
    ds: xr.Dataset, var: str, core_dims: list[str], a: float = 0.1
) -> xr.Dataset:
    """
    unsharp masking via the laplacian. Returns the input dataset with the calculated laplacian, blurred and sharpened sign
    """
    ds_ = ds.assign(
        laplace=lambda x: xr.apply_ufunc(
            ndimage.laplace,
            x[var],
            input_core_dims=[
                core_dims,
            ],
            output_core_dims=[
                core_dims,
            ],
        )
    )

    ds_ = ds_.assign(blurred=lambda x: x[var] - x["laplace"])
    ds_ = ds_.assign(sharpened=lambda x: x[var] + a * (x["blurred"]))

    return ds_


def gaussian(
    ds: xr.Dataset, var: str, core_dims: list[str], sigma: float = 10.0, a: float = 0.1
) -> xr.Dataset:
    """
    Unsharp masking via a Gaussian filter. Returns the input dataset with the calculated laplacian, blurred and sharpened signals.
    """

    ds_ = ds.assign(
        gaussian=lambda x: xr.apply_ufunc(
            ndimage.gaussian_filter,
            x[var],
            sigma,
            input_core_dims=[
                core_dims,
                [],  # needed for sigma arg
            ],
            output_core_dims=[
                core_dims,
            ],
        )
    )

    ds_ = ds_.assign(blurred=lambda x: x[var] - x["gaussian"])

    ds_ = ds_.assign(sharpened=lambda x: x[var] + a * x["blurred"])

    return ds_
