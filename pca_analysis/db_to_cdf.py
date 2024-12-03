from pca_analysis.code.get_sample_data import get_ids_by_varietal
from database_etl.etl.etl_pipeline_raw import get_data
from pca_analysis.notebooks.experiments.parafac2_pipeline.input_data import (
    InputDataGetter,
)
from pca_analysis.definitions import DB_PATH_UV
import duckdb as db

from pca_analysis.notebooks.experiments.parafac2_pipeline.parafac2_decomposition import (
    get_input_data,
)
import polars as pl
import xarray as xr
import matplotlib.pyplot as plt
from database_etl.etl.etl_pipeline_raw import (
    smooth_numeric_col,
    get_sample_metadata,
    fetch_imgs,
)
from pathlib import Path


def get_samples_by_id(conn, ids):
    imgs = fetch_imgs(con=conn, runids=ids)
    metadata_df = get_sample_metadata(con=conn)

    return imgs, metadata_df


def img_tbl_to_xr_arr(imgs: list[pl.DataFrame]):
    runids = [img.get_column("runid").unique().item() for img in imgs]
    mins_labels = imgs[0].get_column("mins").to_numpy(writable=True)
    wavelength_labels = [
        int(col)
        for col in imgs[0].columns
        if col not in ["id", "mins", "runid", "path"]
    ]

    coords = [
        ("runid", runids),
        ("mins", mins_labels),
        ("wavelength", wavelength_labels),
    ]

    import numpy as np

    data_array = np.stack(
        [
            img.drop("id", "mins", "runid", "path").to_numpy(writable=True)
            for img in imgs
        ]
    )

    xr_arr = xr.DataArray(data=data_array, coords=coords)

    return xr_arr


def shiraz_as_xarr(
    db_path: Path,
):
    """ """
    with db.connect(db_path) as conn:
        ids = get_ids_by_varietal("shiraz", conn)
        imgs, metadata_df = get_samples_by_id(conn=conn, ids=ids)

    # todo: add metadata_df to xarr
    img_arr = img_tbl_to_xr_arr(imgs)

    return img_arr


def shiraz_to_cdf(
    db_path: Path = DB_PATH_UV,
    out_path: Path = Path.cwd() / "out_file.nc",
):
    """
    Output the Shiraz dataset as a netcdf file. `db_path` and `out_path` are optional.

    Currently only contains the image data and runids. Adding extra metadata is a TODO.

    Further Reading
    ===============

    - cdf in matlab: <https://au.mathworks.com/help/matlab/network-common-data-form.html>
    - cdf files: <https://www.unidata.ucar.edu/software/netcdf/>
    - xarray to netdf: https://docs.xarray.dev/en/stable/generated/xarray.Dataset.to_netcdf.html
    """
    shiraz_xarr = shiraz_as_xarr(db_path=db_path)

    # remove sample 82 as its an outlier compared ot the other shiraz.
    filtered_xr_arr = shiraz_xarr.where(shiraz_xarr.runid != "82", drop=True)

    filtered_xr_arr.to_dataset(name="shiraz_dataset").to_netcdf(path=out_path)


if __name__ == "__main__":w
    shiraz_to_cdf()
