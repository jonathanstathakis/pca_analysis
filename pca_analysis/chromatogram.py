from copy import deepcopy, copy
import xarray as xr

from pca_analysis import peak_picking
from pca_analysis.peak_picking import get_peak_table_as_df
from pca_analysis.validation import validate_da


class ChromDS:
    @classmethod
    def load_shz_dset(cls):
        """
        load the raw shiraz dataset used for dev.
        """
        from .get_dataset import get_shiraz_dataset

        return ChromDS(get_shiraz_dataset(), input_data="input_data")

    def __init__(self, ds: xr.Dataset, input_data: str):
        """
        wrapper for a spectro-chromatogram represented internally as a dataarray.

        :param: input_data:
            the core input signal of the dataset. This is treated as the base for all downstream transformations and viz.
        """

        self._ds = ds
        self._input_data = input_data
        self.viewer = Viewer(ds=ds, input_data=self._input_data)

    def subset_by_idx(self, subset_da: xr.DataArray):
        """
        Take a dataaray to subset the internal dataarray. Useful for exploratory
        subsetting through the xarray API. Only useful for subsetting.
        """

        self._ds = self._ds.sel(**subset_da.indexes)

    def sel(self, **kwargs):
        _ds = copy(self._ds)
        _ds = _ds.sel(**kwargs)
        return ChromDS(_ds, input_data=self._input_data)

    def isel(self, **kwargs):
        _ds = copy(self._ds)
        _ds = _ds.isel(**kwargs)
        return ChromDS(_ds, input_data=self._input_data)

    def _repr_html_(self):
        return self._ds._repr_html_()


DIMORDER = ["sample", "wavelength", "mins"]


def da_heatmap(da: xr.DataArray):
    validate_da(da, DIMORDER)
    return (
        da.transpose("sample", "mins", "wavelength")
        .plotly.facet(
            x="wavelength", y="mins", z="sample", n_cols=3, plot_type="heatmap"
        )
        .update_layout(height=1000)
    )


class Viewer:
    def __init__(self, ds: xr.Dataset, input_data: str):
        """
        Viz namespace.
        """
        self._input_data = input_data
        self._ds = ds

    def heatmap(self, var=None):
        """
        defaults to 'input_data' if `var=None`
        """

        if var:
            da = self._ds[var]
        else:
            da = self._ds[self._input_data]

        return da_heatmap(da=da)

    def app_2D(self):
        """
        2D app overlaying various variables.
        """

        from .chromviewer import chromviewer_2D

        return chromviewer_2D(data=self._ds)


def ds_plot_peaks(ds: xr.Dataset, peak_outlines, peak_width_calc):
    # ensure that the expected arrays are present
    req_arrs = set(["input_data", "pt"])

    if not req_arrs.issubset(set(ds.keys())):
        raise ValueError(f"expect {req_arrs} to be in ds")

    # ensure that the data is 1 dimensional

    if len(ds["input_data"].sizes.keys()) > 1:
        raise ValueError("can only viz 1D signals")

    peak_table = ds.pt.pipe(get_peak_table_as_df)

    return peak_picking.plot_peaks(
        peak_table=peak_table,
        input_signal=ds.input_data.data.squeeze(),
        peak_outlines=peak_outlines,
        peak_width_calc=peak_width_calc,
    )


# TODO add peak picking.
