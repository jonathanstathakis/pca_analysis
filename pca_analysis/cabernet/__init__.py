from dataclasses import dataclass


class AbstChrom:
    """
    provides a common *chromatogram* validation via the dims - a
    chromatogram should have `SAMPLE`, `TIME`, `SPECTRA` dims only.
    """

    @property
    def CHROM(self):
        return names.CHROM

    @property
    def SAMPLE(self):
        return chrom_dims.SAMPLE

    @property
    def TIME(self):
        return chrom_dims.TIME

    @property
    def SPECTRA(self):
        return chrom_dims.SPECTRA

    @property
    def DIMS(self):
        return chrom_dims.SAMPLE, chrom_dims.TIME, chrom_dims.SPECTRA

    @property
    def CORE_DIM(self):
        return chrom_dims.TIME


@dataclass
class ChromDims:
    SAMPLE: str
    TIME: str
    SPECTRA: str

    def __len__(self):
        return len(self.__dict__.keys())


@dataclass
class Names:
    """
    abstracted DataArray names such as chromatogram, baseline etc.
    """

    CHROM = "input_data"


chrom_dims = ChromDims(SAMPLE="sample", TIME="mins", SPECTRA="wavelength")
names = Names()
