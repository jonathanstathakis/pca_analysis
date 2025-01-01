"""
Provide accessors for analysing chromatographic datasets, using DataArray as the key object.
"""

import xarray as xr
from . import xr_plotly  # noqa
from .validation import validate_da
from . import prepro  # noqa
from . import peak_picking


DIMORDER = ["id_rank", "wavelength", "mins"]


def _get_da_as_df(da: xr.DataArray):
    """return the dataarray as a dataframe with only the key dimensional variables and the data."""
    drop_vars = set(da.coords).difference(set(da.sizes.keys()))

    return da.drop_vars(drop_vars).to_dataframe()


def get_peak_table_as_df(pt: xr.DataArray):
    """
    return the peak table as a flattened, pivoted dataframe
    """

    pt_ = _get_da_as_df(da=pt).unstack().droplevel(0, axis=1).reset_index().dropna()
    return pt_


@xr.register_dataset_accessor("agt")
class AgiletteDS:
    def __init__(self, ds: xr.Dataset):
        """
        Dataset level accessor for Agilette namespace.
        """

        self._ds = ds

    def plot_peaks(self):
        """
        viz peaks as markers overlaying the signal. Dataset must have 'input_data' and 'pt' DataArrays.
        """

        # ensure that the expected arrays are present
        req_arrs = set(["input_data", "pt"])

        if not req_arrs.issubset(set(self._ds.keys())):
            raise ValueError(f"expect {req_arrs} to be in ds")

        # ensure that the data is 1 dimensional

        if len(self._ds["input_data"].sizes.keys()) > 1:
            raise ValueError("can only viz 1D signals")

        peak_table = self._ds.pt.pipe(get_peak_table_as_df)

        return peak_picking.plot_peaks(
            peak_table=peak_table,
            input_signal=self._ds.input_data.data.squeeze(),
            # peak_outlines=False,
            # peak_width_calc=False,
        )


@xr.register_dataarray_accessor("agt")
class AgiletteDA:
    def __init__(self, da: xr.DataArray):
        """
        Chromatogram analyser. For new data, run `preprocessor` first.
        """

        self._da = da

    def preprocess(self):
        return self._da.preprocessor.preprocess_pipe()

    def heatmap(self):
        """
        Generate a faceted heatmap of the data
        """
        validate_da(self._da, DIMORDER)

        return (
            self._da.transpose("id_rank", "mins", "wavelength")
            .plotly.facet(
                x="wavelength", y="mins", z="id_rank", n_cols=3, plot_type="heatmap"
            )
            .update_layout(height=1000)
        )

    def tabulate_peaks(
        self,
        find_peaks_kwargs=dict(),
        peak_widths_kwargs=dict(rel_height=0.95),
    ) -> xr.DataArray:
        return peak_picking.compute_dataarray_peak_table(
            xa=self._da,
            core_dim="mins",
            find_peaks_kwargs=find_peaks_kwargs,
            peak_widths_kwargs=peak_widths_kwargs,
        )
