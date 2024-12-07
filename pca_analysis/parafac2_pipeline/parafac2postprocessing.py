import logging

import numpy as np
import polars as pl
from numpy.typing import NDArray
from tensorly.parafac2_tensor import Parafac2Tensor, apply_parafac2_projections

pl.Config.set_tbl_width_chars(999)
logger = logging.getLogger(__name__)


class Parafac2PostProcessor:
    def __init__(self, decomp: Parafac2Tensor):
        self.A, self.Bs, self.C = apply_projections(decomp)
        self.n_components: int = self.A.shape[1]
        self.n_samples: int = self.A.shape[0]
        self.n_time_points: int = self.Bs[0].shape[0]
        self.n_wavelengths: int = self.C.shape[0]

    def get_component_tensors(self):
        return _construct_sample_component_tensors(A=self.A, Bs=self.Bs, C=self.C)

    def get_component_tensor_df(self):
        dfs = []
        for sample_idx, slice in enumerate(self.get_component_tensors()):
            for cc, component in enumerate(slice):
                dfs.append(_create_component_tensor_df(component, sample_idx, cc))
        return pl.concat(dfs)

    def combine_parafac2_results_input_signal(
        self,
        runid_order,
        input_signal,
    ):
        return combine_parafac2_results_input_signal(
            runid_order=runid_order,
            input_signal=input_signal,
            component_tensors=self.get_component_tensor_df(),
        )

    # def create_reconstruction_slices(self):


def _create_component_tensor_df(
    slice: NDArray, sample_idx: int, component_idx: int
) -> pl.DataFrame:
    """
    create a component specific slice for a given sample_idx and component_idx

    :param slice: the component image
    :type slice: NDArray
    :param sample_idx: the sample index of the sample
    :type sample_idx: int
    :param component_idx: the component index of the component
    :type component_idx: int
    :raises ValueError: if duplicate labels exist in the normalised table
    :return: a normalised component dataframe
    :rtype: pl.DataFrame
    """
    df = (
        pl.DataFrame(slice)
        .with_columns(
            pl.lit(sample_idx).alias("sample"), pl.lit(component_idx).alias("component")
        )
        .with_row_index("elution_point")
        .unpivot(
            index=["sample", "component", "elution_point"],
            variable_name="wavelength",
            value_name="abs",
        )
        .with_columns(pl.col("wavelength").str.replace("column_", "").cast(int))
        .select(
            [
                "sample",
                "component",
                "wavelength",
                "elution_point",
                "abs",
            ]
        )
    )

    dups = df.select(
        "sample", "component", "wavelength", "elution_point"
    ).is_duplicated()

    if dups.any():
        raise ValueError(
            f"duplicate primary key entry detected in sample: {sample_idx}, component: {component_idx}:{df.filter(dups)}"
        )
    return df


def _construct_sample_component_tensors(
    A: NDArray, Bs: list[NDArray], C: NDArray
) -> list[NDArray]:
    """
    for the weights `A`, samplewise elution profiles `Bs` and spectral profile `C`
    create a three mode tensor of each component slice of each sample. Thus accessing
    the component mode will give you the image of that component for each sample.

    - A the weight of each component in each sample with shape: (samples, components)
    - Bs are a list of elution profiles of each component of each sample with shape:
    samples, (elution points, components)
    - C is the spectral profile of each component with shape (spectral points,
    components)

    Returns a sample-wise list (i) of 3 mode tensors (r, j, k). For a 
    spectro-chromatographic dataset that would be samples x component x time x spectra.

    :param A: The weights of the components per sample.
    :type A: NDArray
    :param Bs: The samplewise component elution profiles
    :type Bs: list[NDArray]
    :return: list of 3 mode tensors sample: (components, elution, spectral)
    :rtype: list[NDArray]
    """
    # construct each component of each sample as a np arr

    component_tensors = []
    for sample_idx, B in enumerate(Bs):
        tensor = np.einsum("ik,jk->kij", B * A[sample_idx], C)
        component_tensors.append(tensor)

    return [np.asarray(x) for x in component_tensors]


def apply_projections(
    parafac2tensor: Parafac2Tensor,
) -> tuple[NDArray, list[NDArray], NDArray]:
    """
    apply the tensorly `apply_parafac2_projections` to scale the Bs

    :param parafac2tensor: result of PARAFAC2 decomposition
    :type parafac2tensor: Parafac2Tensor
    :return: A, B slices and C
    :rtype: tuple[NDArray, list[NDArray], NDArray]
    """
    weights, (A, B, C), projections = parafac2tensor
    _, (_, Bs, _) = apply_parafac2_projections((weights, (A, B, C), projections))

    return A, Bs, C


def combine_parafac2_results_input_signal(runid_order, input_signal, component_tensors):
    adapted_runids = _adapt_runids(runid_order)
    adapted_input_signals = _adapt_input_signals(input_signal)
    wavelength_labels = _get_wavelength_labels(input_signals=adapted_input_signals)
    component_tensors = _adapt_component_tensors(
        component_tensors=component_tensors,
        runids=adapted_runids,
        wavelength_labels=wavelength_labels,
    )

    rectified_df = pl.concat(
        [adapted_input_signals, component_tensors], how="vertical_relaxed"
    )

    return rectified_df


def _adapt_runids(runids):
    return runids.rename({"runid_idx": "sample_idx"})


def _adapt_component_tensors(component_tensors, runids, wavelength_labels):
    return (
        component_tensors.rename(
            {
                "sample": "sample_idx",
                "wavelength": "wavelength_idx",
            }
        )
        .with_columns(
            pl.col("sample_idx").cast(pl.UInt32),
            pl.col("wavelength_idx").cast(pl.UInt32),
            (pl.lit("component_") + pl.col("component").cast(str)).alias("signal"),
        )
        .join(runids, on="sample_idx")
        .join(wavelength_labels, on="wavelength_idx")
        .sort(["sample_idx", "component", "wavelength", "elution_point"])
        .select(
            [
                "sample_idx",
                "runid",
                "signal",
                "component",
                "wavelength_idx",
                "wavelength",
                "elution_point",
                "abs",
            ]
        )
    )


def _adapt_input_signals(input_signals):
    return input_signals.rename({"nm": "wavelength", "idx": "elution_point"}).select(
        pl.col("runid").rank("dense").sub(1).alias("sample_idx"),
        "runid",
        pl.lit("input").alias("signal"),
        pl.lit(None).alias("component"),
        pl.col("wavelength").rank("dense").alias("wavelength_idx"),
        "wavelength",
        pl.col("elution_point").cast(pl.UInt32),
        "abs",
    )


def _get_wavelength_labels(input_signals: pl.DataFrame):
    return input_signals.select(
        pl.col("wavelength").unique(),
    ).with_columns(pl.col("wavelength").rank("dense").alias("wavelength_idx"))
