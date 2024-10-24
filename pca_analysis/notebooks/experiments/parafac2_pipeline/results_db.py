import duckdb as db
from .estimators import BCorr_ASLS, PARAFAC2
from .core_tables import CoreTablesDB, CoreTbls
from .parafac2db import PARAFAC2DB, Parafac2Tables
from .bcorrdb import BCorrDB, BcorrTbls
from .pipeline import PipeSteps
import logging

logger = logging.getLogger(__name__)


class ResultsDB:
    def __init__(self, conn: db.DuckDBPyConnection):
        """the results database"""

        self._conn = conn
        self.core_tables = CoreTablesDB(conn=self._conn)
        self.parafac2_db = PARAFAC2DB(output_conn=self._conn)
        self.bcorrdb = BCorrDB(output_conn=self._conn)

        self.core_tbls = CoreTbls
        self.parafac2_tbls = Parafac2Tables
        self.bcorr_tbls = BcorrTbls

    def load_results(
        self,
        exec_id,
        runids,
        steps,
        pipeline,
    ):
        logger.debug("loading results..")

        logger.debug("loading core tables..")

        cl = self.core_tables.get_loader(
            exec_id=exec_id,
            runids=runids,
        )

        cl.load_core_tables()

        if steps == "all":
            steps = [PipeSteps.BCORR, PipeSteps.PARAFAC2]

        if str(PipeSteps.BCORR) in steps:
            bcorr_est: BCorr_ASLS = pipeline.named_steps[PipeSteps.BCORR]
            bc_loader = self.bcorrdb.get_loader(
                exec_id=exec_id,
                baselines=bcorr_est.bline_slices_,
                corrected=bcorr_est.Xt,
            )

            bc_loader.load_results()

        logger.debug("bcorr results ETL complete.")

        if str(PipeSteps.PARAFAC2) in steps:
            parafac2_est: PARAFAC2 = pipeline.named_steps[PipeSteps.PARAFAC2]

            loader = self.parafac2_db.get_loader(
                exec_id=exec_id,
                decomp=parafac2_est.decomp_,
            )

            loader.create_datamart()

        logger.debug("results loading complete.")

    def _clear_database(self):
        """call to delete all rows of all tables of the database"""

        # clear parafac2 tables
        self.parafac2_db.clear_tables()

        # clear bcorr tables
        self.bcorrdb.clear_tables()

        # clear core tables
        self.core_tables.clear_tables()

    def _get_all_table_names(self):
        table_names = (
            list(self.bcorr_tbls) + list(self.core_tbls) + list(self.parafac2_tbls)
        )

        if not len(table_names) == len(set(table_names)):
            raise ValueError("duplicate tables names detected. Check table name enums")
        return table_names

    def _get_database_report(self):
        counts = []
        for table in self._get_all_table_names():
            count = self._conn.execute(
                f"""
                select '{table}'     as table, count(*) as count from {table}
                """
            ).pl()

            counts.append(count)

        import polars as pl

        counts = pl.concat(counts)

        return counts
