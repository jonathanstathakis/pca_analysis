from .. import AbstChrom

from xarray import DataArray, DataTree


class BCorr(AbstChrom):
    def __init__(self, da: DataArray):
        """
        baseline correction namespace.
        """

        self._da = da

    def snip(self, **kwargs):
        """
        TODO: convert to a scikit learn transformer if poss to reduce redundancy.
        """
        from pca_analysis.preprocessing.bcorr import snip as _snip

        ds = _snip(da=self._da, core_dim=self.CORE_DIM, **kwargs)

        dt = DataTree(ds)
        dt.name = "bcorred"
        dt.attrs["data_model_type"] = "baseline corrected"

        from pca_analysis.cabernet.cabernet import Cabernet

        return Cabernet.from_tree(dt=dt)


class Transform:
    def __init__(self, da: DataArray):
        """
        Shiraz transform operation namespace. transform the internal DataArray according
        to some predefined function, returning a Dataset or DataArray.
        """
        self._da = da
        self.bcorr = BCorr(da=self._da)

    def unfold(self, row_dims: tuple[str, str], column_dim: str, new_dim_name: str):
        """
        unfold the internal dataarray along one dimension (the first value of `row_dims`)
        leaving `column_dim` as columns and `row_dims` as augmented dimension of rows.

        TODO how to handle assignment back to cab?
        """

        from ...transformers import Unfolder

        unfolder = Unfolder(
            row_dims=row_dims, column_dim=column_dim, new_dim_name=new_dim_name
        )
        unfolded = unfolder.fit_transform(X=self._da)  # type: ignore

        return unfolded
