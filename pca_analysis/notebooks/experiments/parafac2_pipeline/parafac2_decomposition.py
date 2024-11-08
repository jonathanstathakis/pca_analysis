import logging
from pathlib import Path

import polars as pl

from pca_analysis.notebooks.experiments.parafac2_pipeline.data import Data
from pca_analysis.notebooks.experiments.parafac2_pipeline.input_data import (
    InputDataGetter,
)
from pca_analysis.notebooks.experiments.parafac2_pipeline.pipeline import (
    create_pipeline,
)
from pca_analysis.notebooks.experiments.parafac2_pipeline.pipeline_defs import DCols
from tests.test_definitions import TEST_DB_PATH
from pca_analysis.definitions import ROOT

from pca_analysis.notebooks.experiments.parafac2_pipeline.parafac2postprocessing import (
    Parafac2PostProcessor,
)

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)


def get_data_run_pipeline(input_db_path, runids, filter_expr, parafac2_params={}):
    input_data = get_input_data(input_db_path, runids, filter_expr)

    X = input_data.to_X()

    # use defaults if none provided
    if parafac2_params:
        pipeline = parafac2_decomp(X, parafac2_params)
    else:
        pipeline = parafac2_decomp(X)

    df = postprocess_parafac2(pipeline, runids=X.runids, input_tbl=input_data._nm_tbl)
    return df


def get_input_data(input_db_path, runids, filter_expr=None):
    raw_data_extractor = InputDataGetter(input_db_path=input_db_path, ids=runids)
    input_data = Data(
        time_col=str(DCols.TIME),
        runid_col=str(DCols.RUNID),
        nm_col=str(DCols.NM),
        abs_col=str(DCols.ABS),
        scalar_cols=[str(DCols.PATH), str(DCols.ID)],
    )  # ignore

    input_data = input_data.load_data(raw_data_extractor)

    if filter_expr:
        input_data = input_data.filter_nm_tbl(expr=filter_expr)
    return input_data


def parafac2_decomp(
    X,
    parafac2_params=dict(
        parafac2__nn_modes="all",
        bcorr__lam=1e5,
        parafac2__linesearch=False,
        parafac2__rank=12,
    ),
):
    logfile = Path(ROOT) / "pipeline_log"

    pipeline = create_pipeline()
    pipeline.set_params(**parafac2_params)

    # to capture print for logs. See <https://johnpaton.net/posts/redirect-logging/>
    import contextlib

    with open(logfile, "w") as h, contextlib.redirect_stdout(h):
        pipeline.fit_transform(X.data)

    # display last two lines of the fit report (PARAFAC2)
    with open(logfile, "r") as f:
        logger.info("\n".join(f.readlines()[-2:]))

    return pipeline


def postprocess_parafac2(pipeline, runids, input_tbl):
    pp = Parafac2PostProcessor(decomp=pipeline.named_steps["parafac2"].decomp_)

    rectified_df = pp.combine_parafac2_results_input_signal(
        runid_order=runids, input_signal=input_tbl
    )

    return rectified_df
