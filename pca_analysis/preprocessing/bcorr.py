import numpy as np
from pybaselines.smooth import snip as pysnip
import xarray as xr
from sklearn_xarray.preprocessing import Resampler, BaseTransformer
from xarray import Dataset


def snip_xr(
    ds: xr.Dataset,
    var: str,
    max_half_window=None,
    decreasing=False,
    smooth_half_window=None,
    filter_order: int = 2,
    x_data: str = "",
    **pad_kwargs,
):
    """
    apply SNIP to an input data array

    docs: https://pybaselines.readthedocs.io/en/latest/api/pybaselines/smooth/index.html

    TODO change to `apply_ufunc`
    """
    blines = []
    for x in ds[var]:
        bline, _ = pysnip(
            x,
            max_half_window=max_half_window,
            decreasing=decreasing,
            smooth_half_window=smooth_half_window,
            filter_order=filter_order,
            x_data=ds[x_data],
            pad_kwargs=pad_kwargs,
        )
        blines.append(bline)

    blines_ = np.stack(blines)
    blines_ds = ds[var].copy(data=blines_)
    blines_ds.name = "baselines"

    ds_ = xr.merge([ds, blines_ds])
    ds_ = ds_.assign(data_corr=lambda x: x[var] - x["baselines"])
    return ds_


def apply_snip(x, **kwargs):
    result, _ = pysnip(x, **kwargs)
    return result


class SNIP(BaseTransformer):
    def __init__(
        self,
        core_dim,
        max_half_window=None,
        decreasing: bool = False,
        smooth_half_window=None,
        filter_order: int = 2,
        x_data=None,
        **pad_kwargs,
    ):
        self.core_dim = core_dim
        self.max_half_window = max_half_window
        self.decreasing = decreasing
        self.smooth_half_window = smooth_half_window
        self.filter_order = filter_order
        self.x_data = x_data
        self.pad_kwargs = pad_kwargs

    def fit(self, X, y=None, **fit_params):
        super(SNIP, self).fit(X, y, **fit_params)
        return self

    def transform(self, X, y=None):
        self.Xt, self.baselines_ = snip(da=X, core_dim=self.core_dim)
        return self.Xt


def snip(da: xr.DataArray, core_dim, **kwargs):
    """
    Correct baseline over all samples and wavelengths, adding the baseline
    and corrected signal as variables to the dataset.

    Hardcoded keys
    """

    if not isinstance(da, xr.DataArray):
        raise TypeError

    default_kwargs = dict(
        max_half_window=None,
        decreasing=False,
        smooth_half_window=None,
        filter_order=2,
        x_data=None,
        pad_kwargs=None,
    )

    # merge input kwargs and default kwargs, favoring input kwargs
    merged_kwargs = {}
    for k in default_kwargs:
        if k in kwargs:
            merged_kwargs[k] = kwargs[k]
        else:
            merged_kwargs[k] = default_kwargs[k]

    baselines = xr.apply_ufunc(
        apply_snip,
        da,
        kwargs=merged_kwargs,
        input_core_dims=[
            [core_dim],
        ],
        output_core_dims=[[core_dim]],
        # need vectorize to do the looping
        vectorize=True,
    )

    corr = da - baselines

    corr = corr.rename("corrected")
    baselines = baselines.rename("baselines")
    ds = Dataset(data_vars={corr.name: corr, baselines.name: baselines})
    ds = ds.assign_attrs(bline_fit_params=merged_kwargs)
    return ds
