from .estimators import BCorr_ASLS, PARAFAC2
from .core_tables import CoreTablesDB, CoreTbls
from .parafac2db import Parafac2Tables
from .bcorrdb import BCorrDB, BcorrTbls
from .pipeline import PipeSteps
import logging
import polars as pl

logger = logging.getLogger(__name__)
from sqlalchemy import Engine, text
from .parafac2db import PARAFAC2DB
from sklearn.pipeline import Pipeline


class ResultsDB:
    def __init__(self, engine: Engine):
        """the results database

        TODO: swap to ORM
        """
        self._engine = engine
        self.parafac2db = PARAFAC2DB(engine=self._engine)

        self.core_tables = CoreTablesDB(engine=self._engine)
        # self.parafac2_db = PARAFAC2DB(output_conn=self._conn)
        self.bcorrdb = BCorrDB(engine=self._engine)

        self.core_tbls = CoreTbls
        self.parafac2_tbls = Parafac2Tables
        self.bcorr_tbls = BcorrTbls

    def load_new_results(
        self,
        exec_id: str,
        runids: list[str],
        steps: list[str],
        pipeline: Pipeline,
        wavelength_labels: list[int],
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
            bc_loader = self.bcorrdb.get_loader()
            bc_loader.load_results(
                exec_id=exec_id,
                baselines=bcorr_est.bline_slices_,
                corrected=bcorr_est.Xt,
                runids=runids,
                wavelength_labels=wavelength_labels,
            )

        logger.debug("bcorr results ETL complete.")

        if str(PipeSteps.PARAFAC2) in steps:
            parafac2_est: PARAFAC2 = pipeline.named_steps[PipeSteps.PARAFAC2]

            loader = self.parafac2db.get_loader(
                exec_id=exec_id,
                decomp=parafac2_est.decomp_,
                runids=runids,
                wavelength_labels=wavelength_labels,
            )

            loader.create_datamart()

        logger.debug("results loading complete.")

    def _get_all_table_names(self):
        table_names = (
            pl.read_database(
                "select distinct table_name from information_schema.tables",
                self._engine,
            )
            .get_column("table_name")
            .to_list()
        )
        return table_names

    def _get_database_report(self):
        counts = []

        for table in self._get_all_table_names():
            with self._engine.connect() as conn:
                query = f"""
                    select '{table}' as table, count(*) as count from {table}
                    """
                count = pl.read_database(query, conn)

            counts.append(count)
        counts = pl.concat(counts)

        return counts
