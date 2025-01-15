from pca_analysis.cabernet import AbstChrom
from ...transformers import Unfolder
from ...decomposers import PCA
from sklearn.pipeline import Pipeline
from xarray import DataArray


class Decomposition(AbstChrom):
    def __init__(self, da: DataArray):
        self._da = da

    def pca(self, standardscale: bool = False):
        """
        perform PCA on the input dataarray. Thus particular implementation requires that the input array is 2D thus
        it unfolds along the sample domain.

        standardscale : bool, default = False
            whether to include a standard scaling stage to the PCA pipeline.

        """

        pca_pipeline = Pipeline(
            [
                (
                    "unfold",
                    Unfolder(
                        row_dims=(
                            self.SAMPLE,
                            self.TIME,
                        ),
                        column_dim=self.SPECTRA,
                        new_dim_name="aug",
                    ),
                ),
                ("pca", PCA()),
            ]
        )

        if standardscale:
            from sklearn.preprocessing import StandardScaler

            pca_pipeline.steps.insert(1, ("scaler", StandardScaler()))

        pca_pipeline.fit_transform(X=self._da)

        pca = pca_pipeline.named_steps["pca"]

        assert isinstance(pca, PCA)
        return pca
