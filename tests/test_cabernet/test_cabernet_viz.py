import pytest
from pca_analysis.exp_manager.cabernet import Cabernet, Shiraz
from xarray import Dataset, DataArray, DataTree


@pytest.fixture
def cab_2_vars(cab: Cabernet):
    input_data = cab["input_data"]
    if not isinstance(input_data, Shiraz):
        raise TypeError

    cab["data_trans"] = input_data._da - 10 * input_data._da.diff(dim="mins")

    cab = cab.isel(sample=slice(0, 4), wavelength=slice(0, 106, 40))

    cab = cab.sel(mins=slice(0, 5))

    return cab


def test_cab_line_multiple_vars(cab_2_vars: Cabernet):
    fig = cab_2_vars.viz.line_multiple_vars(
        vars=["input_data", "data_trans"],
        x="mins",
        col_wrap=2,
        line_dash_dim="wavelength",
        color_dim="vars",
        facet_dim="sample",
    )

    fig.show()
