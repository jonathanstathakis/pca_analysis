from database_etl.etl.etl_pipeline_raw import get_data
import duckdb as db
from collections import UserList
import polars as pl


class InputData(UserList):
    def __init__(self, con: db.DuckDBPyConnection, ids: list[str]):
        """a wrapper around the list data container for validation. Essentially wraps
        database_etl's `get_data`

        TODO: add validation
        """

        self.data = get_data(output="tuple", con=con, runids=ids)

    def to_long_tables(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Convert InputData to two long polars dataframe tables, one for the image data
        and the second for the sample metadata

        :return: a two element tuple of dataframes. the first is the images and the
        second is the sample metadata
        :rtype: tuple[pl.DataFrame, pl.DataFrame]
        """

        imgs = pl.concat([tup[0] for tup in self.data])
        mta = pl.concat([tup[1] for tup in self.data])

        return imgs, mta
