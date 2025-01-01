"""
IO validation tools.
"""


def validate_da(da, expected_dims: list[str]):
    """
    ensure that input dataarray matches expectation.
    """

    if list(da.sizes.keys()) != expected_dims:
        raise ValueError(
            f"Expecting a DataArray with 3 dims: {expected_dims}, got {list(da.sizes.keys())}"
        )
