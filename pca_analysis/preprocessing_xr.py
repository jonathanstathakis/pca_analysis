"""
provide xr accessors for signal preprocessing functions.
"""

import xarray as xr
from pca_analysis.preprocessing import unsharp_mask


@xr.register_dataset_accessor("preproc")
class PreprocessingDS:
    def __init__(self, ds):
        """
        Namespace providing methods of signal preprocessing. Includes sharpening via
        unsharp masking, smoothing via savgol filter, baseline correction via SNIP and
        partitioning through HCA.
        """
        self._ds = ds
        self.unsharp = Unsharp(ds)
        self.bcorr = BCorr(ds)


class Unsharp:
    def __init__(self, ds):
        """
        Sharpen signals via unsharp masking. This namespace provides access to
        unsharp masking via either the Laplacian or a Gaussian filter.
        """
        self._ds = ds

    def laplacian(self, var: str, core_dims: list[str], a: float = 0.1):
        """
        unsharp masking via the laplacian. Returns the input dataset with the calculated
        laplacian, blurred and sharpened arrays.
        """
        return unsharp_mask.laplacian(ds=self._ds, var=var, core_dims=core_dims, a=a)

    def gaussian(
        self, var: str, core_dims: list[str], sigma: float = 10.0, a: float = 0.1
    ):
        """
        unsharp masking via a Gaussuan filter. Returns the input dataset with the
          calculated Gaussian, blurred and sharpened arrays.
        """

        return unsharp_mask.gaussian(
            ds=self._ds, var=var, core_dims=core_dims, sigma=sigma, a=a
        )


class BCorr:
    def __init__(self, ds: xr.Dataset):
        """
        Baseline Correction
        """
        self._ds = ds

    def snip(self, **kwargs):
        """
        baseline correction via SNIP.
        """
        from pca_analysis.preprocessing import bcorr

        bcorr.snip(data=self._ds, **kwargs)


class Partition:
    def __init__(self, ds: xr.Dataset):
        """
        Partitioning
        """

        self._ds = ds


@xr.register_dataarray_accessor("preproc")
class PreprocessingDA:
    def __init__(self, da):
        self._da = da

    def smooth(self, **kwargs):
        from pca_analysis.preprocessing import smooth

        smooth.savgol_smooth(self._da, savgol_kwargs=kwargs)
