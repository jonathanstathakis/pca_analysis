import numpy as np
from pybaselines.smooth import snip
import xarray as xr


def snip_xr(
    ds: xr.Dataset,
    **kwargs,
):
    """
    apply SNIP to an input data array

    docs: https://pybaselines.readthedocs.io/en/latest/api/pybaselines/smooth/index.html

    """
    blines = []
    for x in ds["raw_data"]:
        bline, _ = snip(x, **kwargs)
        blines.append(bline)

    blines_ = np.stack(blines)
    blines_ds = ds["raw_data"].copy(data=blines_)
    blines_ds.name = "baselines"

    ds_ = xr.merge([ds, blines_ds])
    ds_ = ds_.assign(data_corr=lambda x: x["raw_data"] - x["baselines"])
    return ds_


def apply_snip(x, **kwargs):
    result, _ = snip(x, **kwargs)
    return result


def snip_ds(ds: xr.Dataset, core_dim, **kwargs):
    """
    Correct baseline over all samples and wavelengths, adding the baseline
    and corrected signal as variables to the dataset.

    Hardcoded keys
    """

    if not isinstance(ds, xr.Dataset):
        raise TypeError

    ds = ds.assign(
        baselines=xr.apply_ufunc(
            apply_snip,
            ds.raw_data,
            kwargs=kwargs,
            input_core_dims=[
                [core_dim],
            ],
            output_core_dims=[[core_dim]],
            # need vectorize to do the looping
            vectorize=True,
        )
    )
    ds = ds.assign(data_corr=ds.raw_data - ds.baselines)

    return ds
