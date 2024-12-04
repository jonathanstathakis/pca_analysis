import duckdb as db
from database_etl import get_data
import numpy as np
import xarray as xr
from pymatreader import read_mat

by_varietal_query = """--sql
            select
                inc_chm.runid
            from
                inc_chm
            join
                st
            using
                (samplecode)
            join
                ct
            using
                (vintage, wine)
            where varietal = ?
    """


def get_ids_by_varietal(varietal: str, con: db.DuckDBPyConnection):
    return (
        con.execute(
            by_varietal_query,
            parameters=[varietal],
        )
        .pl()["runid"]
        .to_list()
    )


def get_shiraz_data(con: db.DuckDBPyConnection):
    shiraz_ids = get_ids_by_varietal(varietal="shiraz", con=con)
    sh_data = get_data(output="tuple", con=con, runids=shiraz_ids)

    return sh_data


def get_zhang_data(path="/Users/jonathan/mres_thesis/pca_analysis/Wine_v7.mat"):
    key_mapping = {
        "sample": "Label_Wine_samples",
        "time": "Label_Elution_time",
        "mz": "Label_Mass_channels",
    }

    data = read_mat(
        filename=path, variable_names=["Data_GC"] + list(key_mapping.values())
    )

    raw_data = xr.DataArray(
        data["Data_GC"],
        coords=[data[k] for k in key_mapping.values()],
        dims=list(key_mapping.keys()),
    )

    idx_start = np.nonzero(np.isclose(raw_data["time"], 16.52, atol=1e-2))[0][0] + -6
    idx_end = np.nonzero(np.isclose(raw_data["time"], 16.76, atol=1e-2))[0][0] + 6

    sliced_data = raw_data.isel(time=slice(idx_start, idx_end))

    return sliced_data
