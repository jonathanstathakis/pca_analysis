from sklearn_xarray.preprocessing import BaseTransformer


def unfold(
    x,
    row_dims: tuple[str, str] = ("sample", "time"),
    column_dim: str = "mz",
    new_dim_name="aug",
):
    return x.stack({new_dim_name: row_dims}).transpose(..., column_dim)


class Unfolder(BaseTransformer):
    def __init__(self, row_dims: tuple[str, str], column_dim: str, new_dim_name: str):
        """
        unfold the internal dataarray along one dimension (the first value of `row_dims`)
        leaving `column_dim` as columns and `row_dims` as augmented dimension of rows.

        Takes advantage of sklearn_xarray to apply sklearn
        estimators to DataArrays.
        """
        # required for API
        self.row_dims = row_dims
        self.column_dim = column_dim
        self.new_dim_name = new_dim_name
        self.groupby = None

    def _transform(self, X):
        unfolded = unfold(
            x=X,
            row_dims=self.row_dims,
            column_dim=self.column_dim,
            new_dim_name=self.new_dim_name,
        )
        return unfolded
