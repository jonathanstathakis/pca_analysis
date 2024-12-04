from tensorly.parafac2_tensor import Parafac2Tensor
from pca_analysis.parafac2_pipeline.estimators import PARAFAC2


def test_parafac2_ft(parafac2_ft: PARAFAC2):
    """test if the PARAFAC2 transformer can transform the input data"""
    assert parafac2_ft


def test_decomp(decomp: Parafac2Tensor):
    """assert that the decomp object can be created"""

    assert decomp
