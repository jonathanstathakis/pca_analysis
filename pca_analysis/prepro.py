"""
xr accessor for preparing the data for ChromAnalPipe.
"""

PREPRO_DIMS = ["id", "mins", "wavelength"]


def preprocess_pipe(da):
    return (
        da.pipe(_rank_id)
        .pipe(_rename_images_to_input_data)
        .pipe(_correct_wavelength_datatype)
        .transpose("sample", "wavelength", "mins")
    )


def _rank_id(da):
    da_ = da.assign_coords(
        sample=lambda x: (
            "id",
            x.coords["id"].to_dataframe()["id"].rank(method="dense").astype(int),
        )
    ).swap_dims({"id": "sample"})

    return da_


def _rename_images_to_input_data(da):
    return da.rename("input_data")


def _correct_wavelength_datatype(da):
    return da.assign_coords(
        wavelength=lambda x: ("wavelength", x["wavelength"].astype(int).data)
    )
