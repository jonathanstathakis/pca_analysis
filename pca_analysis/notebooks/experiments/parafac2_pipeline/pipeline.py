from sklearn.pipeline import Pipeline
from pca_analysis.notebooks.experiments.parafac2_pipeline.estimators import (
    PARAFAC2,
    BCorr_ASLS,
)


def create_pipeline():
    """Create the sklearn pipeline"""
    return Pipeline(
        steps=[
            ("bcorr", BCorr_ASLS()),
            (
                "parafac2",
                PARAFAC2(),
            ),
        ],
        verbose=True,
    )
