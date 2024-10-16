from sklearn.pipeline import Pipeline
from pca_analysis.notebooks.experiments.parafac2_pipeline.estimators import PARAFAC2


def create_pipeline(rank: int):
    """Create the sklearn pipeline"""
    return Pipeline(
        steps=[
            (
                "parafac2",
                PARAFAC2(rank=rank, verbose=True, nn_modes="all", linesearch=False),
            )
        ],
        verbose=True,
    )
