import pytest
from pca_analysis.cabernet.cabernet import Cabernet
from xarray import DataArray

from pca_analysis.cabernet.shiraz.shiraz import Shiraz
from pca_analysis.cabernet.shiraz.viz import VizShiraz


@pytest.fixture
def cab_for_trans(cab: Cabernet) -> Cabernet:
    """
    A subset of the Cabernet test set, destined to grow with transformations.
    """
    return cab.isel(sample=[0, 2]).sel(wavelength=[256, 330])


@pytest.fixture
def shz_trans(cab_for_trans: Cabernet) -> Shiraz:
    shz = cab_for_trans["input_data"]
    if isinstance(shz, Shiraz):
        return shz
    else:
        raise TypeError


@pytest.fixture
def bcorred(shz_trans: Shiraz):
    """
    Cabernet object with baseline fitted and corrected data, 2 samples.
    """
    return shz_trans.trans.bcorr.snip()


def test_shiraz_bcorr(bcorred: Cabernet):
    """
    test that Shiraz transform its internal dataarray, returning a
    Cabernet of baseline, bcorr.
    """

    assert bcorred is not None

    assert {"corrected", "baselines"}.issubset(list(bcorred.data_vars))


def test_shiraz_bcorr_attrs(bcorred: Cabernet):
    """
    test that the attributes contain the fit parameters.
    """

    # fit parameters
    assert bcorred._dt.attrs["bline_fit_params"]
    assert bcorred._dt.attrs["data_model_type"] == "baseline corrected"


def test_assign_bcorr_to_cabernet(bcorred: Cabernet, cab_for_trans: Cabernet):
    """
    Assign the results of baseline correction to a Cabernet tree.
    """
    name = bcorred.name
    if isinstance(name, str):
        cab_for_trans = cab_for_trans.assign(**{name: bcorred})
        assert cab_for_trans is not None
    else:
        raise TypeError
