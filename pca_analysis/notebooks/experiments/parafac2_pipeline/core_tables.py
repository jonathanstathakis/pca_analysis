from enum import StrEnum
import duckdb as db
import numpy as np
import polars as pl
import logging

logger = logging.getLogger(__name__)


class CoreTbls(StrEnum):
    EXEC_ID = "exec_id"
    RESULT_ID = "result_id"
    SAMPLES = "samples"


class CoreTablesDB:
    def __init__(self, conn: db.DuckDBPyConnection):
        """wrapper around the core tables of the results db, providing IO for those tables"""

        self._conn = conn

    def get_loader(self, exec_id, runids):
        return CoreTableLoader(exec_id=exec_id, runids=runids, conn=self._conn)

    def clear_tables(self):
        """delete all rows of core tables 'exec_id', 'results_id' and 'samples'."""

        for table in CoreTbls:
            self._conn.execute(f"truncate {table}")


class CoreTableLoader:
    def __init__(
        self,
        exec_id: str,
        runids: list[str],
        conn: db.DuckDBPyConnection = db.connect(),
    ):
        """loads the core tables that other loaders depend on"""
        self._exec_id = exec_id
        self._conn = conn
        self._runids = runids

    def load_core_tables(self):
        self._create_exec_id_tbl()
        self.create_result_id_tbl()
        self._create_sample_table(runids=self._runids)

    def _create_exec_id_tbl(self):
        """creates a exec_id table containing the exec_ids"""

        query = """--sql
        create table if not exists exec_id (
        exec_id varchar primary key,
        );
        insert or replace into exec_id values (?)
        """

        self._conn.execute(query, parameters=[self._exec_id])

    def _create_sample_table(self, runids: list[str]):
        """write a table containing the unique sample ids
        :param runids: the labels of each sample in the dataset
        :type runids: list[str]
        """

        logger.debug("writing sample table..")
        sample_df = pl.DataFrame(
            {"sample": np.arange(0, len(runids), 1), "runid": runids}
        ).with_columns(pl.lit(self._exec_id).alias("exec_id"))
        sample_df.shape

        self._conn.execute("""--sql
            create table if not exists samples (
            exec_id varchar references exec_id(exec_id),
            sample int not null unique,
            runid varchar not null unique,
            primary key (exec_id, sample, runid)
            );
            """)

        self._conn.execute("""--sql
            insert into samples
            select
                exec_id,
                sample,
                runid
            from
                sample_df
            -- this whole table is primary keys, so fundamentally its not up to the 
            -- table to validate those values.
            """)

    def create_result_id_tbl(self):
        """create an empty result_id table containing the identifier for each result type"""

        query = """--sql
        create table if not exists result_id (
        exec_id varchar references exec_id(exec_id),
        result_id varchar unique not null,
        primary key (exec_id, result_id)
        );
        """

        self._conn.execute(query)
