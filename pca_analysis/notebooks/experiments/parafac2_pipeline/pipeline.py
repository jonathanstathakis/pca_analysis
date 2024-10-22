from sklearn.pipeline import Pipeline
from .estimators import (
    PARAFAC2,
    BCorr_ASLS,
)
from enum import StrEnum


class PipeSteps(StrEnum):
    BCORR = "bcorr"
    PARAFAC2 = "parafac2"


def create_pipeline():
    """Create the sklearn pipeline"""
    return Pipeline(
        steps=[
            (PipeSteps.BCORR, BCorr_ASLS()),
            (
                PipeSteps.PARAFAC2,
                PARAFAC2(),
            ),
        ],
        verbose=True,
    )
