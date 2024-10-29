from database_etl.etl.etl_pipeline_raw import get_data
import duckdb as db
from collections import UserList
import polars as pl


class InputDataGetter(UserList):
    def __init__(self, conn: db.DuckDBPyConnection, ids: list[str]):
        """a wrapper around the list data container for validation. Essentially wraps
        database_etl's `get_data`

        TODO: add validation
        """

        self.data: list
        self._conn = conn
        self.ids = ids

    def get_data_as_list_of_tuples(self):
        result = get_data(output="tuple", con=self._conn, runids=self.ids)

        if isinstance(result, list):
            self.data = result
        else:
            raise TypeError

    def as_long_tables(self) -> tuple[pl.DataFrame, pl.DataFrame]:
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
