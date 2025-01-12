"""
dataset fetching functions.
"""

import duckdb as db
import xarray as xr
from database_etl import get_data
from pymatreader import read_mat
from xarray import Dataset

from pca_analysis.definitions import DATA_DIR, DB_PATH_UV
from pca_analysis.get_sample_data import get_ids_by_varietal

from .prepro import preprocess_pipe


def get_shiraz_dataset() -> Dataset:
    with db.connect(DB_PATH_UV) as conn:
        ids = get_ids_by_varietal("shiraz", conn)

        ds = get_data(output="xr", con=conn, runids=ids)

        if not isinstance(ds, Dataset):
            raise TypeError

    return ds.imgs.pipe(preprocess_pipe).to_dataset()


def load_zhang_data():
    mat_path = DATA_DIR / "Wine_v7.mat"

    nc_path = DATA_DIR / "Wine_v7.nc"

    if nc_path.exists():
        full_data = xr.open_dataarray(nc_path)

    else:
        mat_data = read_mat(mat_path)

        key_it = [
            "Label_Wine_samples",
            "Label_Elution_time",
            "Label_Mass_channels",
        ]

        full_data = xr.DataArray(
            mat_data["Data_GC"],
            dims=["sample", "time", "mz"],
            coords=[mat_data[k] for k in key_it],
        )

        full_data.name = "zhang_data"
        full_data.to_netcdf(nc_path)

    return full_data
