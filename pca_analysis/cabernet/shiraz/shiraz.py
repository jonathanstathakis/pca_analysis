from .. import AbstChrom
from xarray import DataArray, Dataset
from .transform import Transform
from .viz import VizShiraz
from .decomposition import Decomposition


class Stats(AbstChrom):
    def __init__(self, da: DataArray):
        self._da = da

    @property
    def mean_max_spectral_label(self):
        return self._da.mean(self.TIME).mean(self.SAMPLE).idxmax().item()

    @property
    def max_sample_label(self):
        return self._da.mean(self.TIME).mean(self.SPECTRA).idxmax().item()


class Shiraz(AbstChrom):
    def __init__(self, da: DataArray):
        """
        Manager for DataArrays
        TODO: add data model type specific transformer and visualiser classes and dispatcher.
        """

        assert isinstance(da, DataArray)

        self._da = da
        self.trans = Transform(da=self._da)

    def __len__(self):
        return self._da.__len__()

    def copy(self):
        """
        make a copy of the internal DataTree, returning a new Cabernet object.
        """

        da = self._da.copy()

        shz = Shiraz(da=da)

        return shz

    def sel(
        self, indexers=None, method=None, tolerance=None, drop=False, **indexers_kwargs
    ):
        shz = self.copy()
        shz._da = shz._da.sel(
            indexers=indexers,
            method=method,
            tolerance=tolerance,
            drop=drop,
            **indexers_kwargs,
        )
        return shz

    def isel(self, **kwargs):
        shz = self.copy()
        shz._da = shz._da.isel(**kwargs)
        return shz

    @property
    def shape(self):
        return self._da.shape

    @property
    def attrs(self):
        return self._da.attrs

    @property
    def viz(self) -> VizShiraz:
        return VizShiraz(da=self._da)

    @property
    def decomp(self):
        return Decomposition(da=self._da)

    @property
    def dims(self):
        return self._da.dims

    @property
    def stats(self):
        return Stats(da=self._da)

    def __getitem__(self, key):
        result = self._da[key]

        if isinstance(result, Dataset):
            return Shiraz(da=result)
        else:
            return result

    def __setitem__(self, key, data):
        self._da[key] = data

        return Shiraz(da=self._da)

    def get(self, key, default=None):
        result = self._da.__get__(name=key, default=default)
        return result

    def __repr__(self):
        return self._da.__repr__()

    def _repr_html_(self):
        return self._da._repr_html_()
