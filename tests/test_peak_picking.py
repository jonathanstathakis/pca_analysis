import pytest
from pca_analysis.peak_picking import PeakPicker
from pca_analysis.cabernet.shiraz.shiraz import Shiraz
from xarray import DataArray
from pca_analysis.get_dataset import get_shiraz_dataset
# TODO define fixtures. input data, peak picker with pick_peaks run.
# TODO define test.


@pytest.fixture
def pick_peaks_data():
    data = (
        get_shiraz_dataset()
        .isel(sample=[0, 1])
        .sel(wavelength=[256, 330])["input_data"]
    )

    if isinstance(data, DataArray):
        return data
    else:
        raise TypeError


@pytest.fixture
def pp(pick_peaks_data: DataArray):
    pp = PeakPicker(da=pick_peaks_data)
    pp.pick_peaks(core_dim="mins", find_peaks_kwargs=dict(rel_height=0.5))

    return pp


def test_peak_table(pp: PeakPicker):
    assert pp.table is not None
    assert not pp.table.empty


def test_dataarray(pp: PeakPicker):
    assert pp.dataarray is not None
    assert sum(pp.dataarray.sizes.values()) > 0
