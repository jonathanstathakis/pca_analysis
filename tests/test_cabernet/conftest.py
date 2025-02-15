import pytest

from xarray import DataArray, Dataset, DataTree
from pca_analysis import get_dataset
from pca_analysis.cabernet.cabernet import Cabernet


@pytest.fixture
def testds() -> Dataset:
    return get_dataset.get_shiraz_dataset()


@pytest.fixture
def cab():
    cab = Cabernet.load_from_dataset("shiraz")
    return cab.assign(child=DataTree(Dataset()))


@pytest.fixture
def shz_input_data(cab: Cabernet):
    return cab["input_data"].sel(mins=slice(0, 30), wavelength=slice(190, 400, 10))
