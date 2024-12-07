from .estimators import BCorr_ARPLS, PARAFAC2
from .bcorrdb import BCorrDB
from .pipeline import PipeSteps
import logging
from .parafac2db import PARAFAC2DB
from sklearn.pipeline import Pipeline
from .core_tables import load_core_tables

logger = logging.getLogger(__name__)


def load_new_results(
    engine,
    exec_id: str,
    runids: list[str],
    steps: list[str],
    pipeline: Pipeline,
    wavelength_labels: list[int],
):
    logger.debug("loading results..")

    logger.debug("loading core tables..")

    load_core_tables(engine, exec_id, runids)

    if steps == "all":
        steps = [PipeSteps.BCORR, PipeSteps.PARAFAC2]

    if str(PipeSteps.BCORR) in steps:
        bcorr_est: BCorr_ARPLS = pipeline.named_steps[PipeSteps.BCORR]

        bcorrdb = BCorrDB(engine=engine)
        bc_loader = bcorrdb.get_loader()

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
        parafac2db = PARAFAC2DB(engine=engine)
        loader = parafac2db.get_loader(
            exec_id=exec_id,
            decomp=parafac2_est.decomp_,
            runids=runids,
            wavelength_labels=wavelength_labels,
        )

        loader.create_datamart()

    logger.debug("results loading complete.")
