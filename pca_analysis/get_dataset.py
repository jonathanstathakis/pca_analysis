"""
dataset fetching functions.
"""

import duckdb as db
from pca_analysis.get_sample_data import get_ids_by_varietal
from pca_analysis.definitions import DB_PATH_UV
from database_etl import get_data
from xarray import Dataset


def get_shiraz_dataset() -> Dataset:
    with db.connect(DB_PATH_UV) as conn:
        ids = get_ids_by_varietal("shiraz", conn)

        ds = get_data(output="xr", con=conn, runids=ids)

        if not isinstance(ds, Dataset):
            raise TypeError

    return ds.imgs.agt.preprocess().to_dataset()
