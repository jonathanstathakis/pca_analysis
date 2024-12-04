"""output a dataset to MatFile"""

# get the test data as two tables: metadata and a samplewise stacked img table
import logging

import duckdb as db

from pca_analysis.get_sample_data import get_ids_by_varietal
from pca_analysis.definitions import DB_PATH_UV
from pca_analysis.parafac2_pipeline.parafac2_decomposition import (
    get_input_data,
)

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)


def dataset_to_matlab(path):
    with db.connect(DB_PATH_UV) as conn:
        ids = get_ids_by_varietal("shiraz", conn)

    input_data = get_input_data(DB_PATH_UV, ids)

    X = input_data.to_X()

    X.to_mat(path)


if __name__ == "__main__":
    dataset_to_matlab(path="x.mat")
