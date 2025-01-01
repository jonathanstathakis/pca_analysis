"""
xr accessor for preparing the data for ChromAnalPipe.
"""

import xarray as xr
from .validation import validate_da

PREPRO_DIMS = ["id", "mins", "wavelength"]


@xr.register_dataarray_accessor("preprocessor")
class Preprocessor:
    def __init__(self, da: xr.DataArray):
        """
        preprocess the data to what ChromAnalPipe expects
        """
        validate_da(da, PREPRO_DIMS)
        self._da = da

    def preprocess_pipe(self):
        """
        All preprocessing steps executed at once
        """

        return self._da.pipe(preprocess_pipe)


def preprocess_pipe(da):
    return (
        da.pipe(_rank_id)
        .pipe(_rename_images_to_input_data)
        .pipe(_correct_wavelength_datatype)
        .transpose("id_rank", "wavelength", "mins")
    )


def _rank_id(da):
    da_ = da.assign_coords(
        id_rank=lambda x: (
            "id",
            x.coords["id"].to_dataframe()["id"].rank(method="dense").astype(int),
        )
    ).swap_dims({"id": "id_rank"})

    return da_


def _rename_images_to_input_data(da):
    return da.rename("input_data")


def _correct_wavelength_datatype(da):
    return da.assign_coords(
        wavelength=lambda x: ("wavelength", x["wavelength"].astype(int).data)
    )
