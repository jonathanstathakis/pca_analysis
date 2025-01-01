"""
Peak picking methods
"""

import xarray as xr
from pca_analysis import peak_picking


@xr.register_dataarray_accessor("peak_picking")
class PeakPicking:
    def __init__(self, da: xr.DataArray):
        """
        provide peak picking methods
        """
        self._da = da

    def tabulate_peaks_1D(
        self, find_peaks_kwargs=dict(), peak_widths_kwargs=dict(rel_height=0.95)
    ):
        """
        Generate a peak table for the DataArray. the da must consist
        of 1 dimension.
        """

        return peak_picking.tablulate_peaks_1D(
            x=self._da.data,
            find_peaks_kwargs=find_peaks_kwargs,
            peak_widths_kwargs=peak_widths_kwargs,
        )
