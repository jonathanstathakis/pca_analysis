import pytest
import xarray as xr
from pca_analysis.cabernet.cabernet import Cabernet
from pca_analysis.cabernet.shiraz.shiraz import Shiraz


def test_Cabernet_init():
    """
    test whether a `Cabernet` object can be initialised
    """
    assert Cabernet(da=xr.DataArray()) is not None


def test_Cabernet_assign():
    """
    test whether a `Cabernet` object can be assigned to, returning an instance of Cabernet.
    """

    cab = Cabernet(da=xr.DataArray()).assign(child=xr.DataTree())

    assert isinstance(cab, Cabernet)


def test_load_from_dataset(cab: Cabernet):
    assert cab
    assert cab._dt


def test_cabernet_getitem_any(cab: Cabernet):
    """
    `__getitem__` returns a `DataArray` or DataTree
    depending on whether the key refers to a
    child node or internal `var`. This test checks
    if the call returns anything.
    """
    input_data = cab["input_data"]
    assert input_data is not None


def test_cabernet_getitem_cab(cab: Cabernet):
    """
    tests if `__getitem__` returns a instance of
    `Cabernet` rather than `DataTree`, as we're
    wrapping `DataTree`.
    """

    assert isinstance(cab["child"], Cabernet), f"{type(cab['child'])}"


def test_cabernet_getitem_shz(cab: Cabernet):
    """
    tests if `__getitem__` returns a instance of `Shiraz` rather than `DataArray` when
     accessing a `DataArray`
    """

    shz = cab["input_data"]

    assert isinstance(shz, Shiraz)


def test_setitem(cab: Cabernet):
    """
    `__setitem__` takes arrays or child nodes. As `Cabernet`
    simply passes it through to the `DataTree`, we can use
    that for validation.
    """

    node = xr.DataTree()

    cab["new_item"] = node

    assert cab.get("new_item") is not None


def test_cab_isel(cab: Cabernet):
    """
    test that isel correctly subsets.
    """
    result = cab.isel(sample=[0, 1])["sample"]
    if isinstance(result, Shiraz):
        assert len(result) == 2
    else:
        raise TypeError


def test_cabernet_set_dim_names(cab: Cabernet):
    """
    test whether the names can be altered in the global space via importing the var.
    """

    from pca_analysis import cabernet

    shz = cab["input_data"]

    if not isinstance(shz, Shiraz):
        raise TypeError

    da = shz._da

    da = da.rename({"wavelength": "mz", "mins": "time"})

    assert set(da.sizes.keys()) == set(["mz", "sample", "time"])

    cabernet.ChromDims.TIME = "time"
    cabernet.ChromDims.SPECTRA = "mz"
    cabernet.ChromDims.SAMPLE = "sample"

    new_cab = Cabernet(da)

    assert new_cab is not None


# TODO establish viz protocols. The basic ones need to be instilled at the Shiraz level. Cabernet can have viz methods but they will be very high level. The idea is that you should directly access the object you want to viz without respect to the higher level structure. Cabernet can have multi-var viz methods such as Comparing input data and baseline corrected data.

# TODO2 establish validation of 'Cabernet', 'Pinot Noir' and 'Shiraz' xarray data objects. That means that the internal xarray object has at least (possibly exclusively) dims SAMPLE, TIME, SPECTRA, as defined by the class level constants. As we define chromatograms to consist of the signal and the peak mapping, we should enforce the DataArray to consist only of these dims, and if a peak map is present, it will have its own constrictions.
