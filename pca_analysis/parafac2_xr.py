"""
Contains functions for working with hplc-uv/vis tensors in xarray and PARAFAC2
decompositions.

tensor dim labels are preset 'COORD1_LABEL' etc. To modify, assign to the corresponding constant.

TODO4 test
"""

from tensorly.decomposition import parafac2
from tensorly.parafac2_tensor import Parafac2Tensor
from tensorly.parafac2_tensor import apply_parafac2_projections
import xarray as xr
import numpy as np

COORD1_LABEL = "sample"
COORD2_LABEL = "time"
COORD3_LABEL = "mz"
A_LABEL = "A"
BS_LABEL = "Bs"
C_LABEL = "C"
COMPONENT_LABEL = "component"


def decomp_as_xr(
    input_data: xr.DataArray,
    rank: int,
    decomp: Parafac2Tensor,
):
    """
    Taking a Parafac2Tensor, convert it to a xr.Dataset using the input xr.
    DataArray. Requires the model rank and coordinate labels.

    Obviously this is not useful the input data was not an xr.DataArray.
    """
    applied_projections = apply_parafac2_projections(decomp)
    A = applied_projections[1][0]
    B = np.stack(applied_projections[1][1])
    C = applied_projections[1][2]

    dim1_coords = input_data.coords[COORD1_LABEL]
    dim2_coords = input_data.coords[COORD2_LABEL]
    rank_labels = [x + 1 for x in range(rank)]
    dim3_coords = input_data.coords[COORD3_LABEL]

    xr_A = xr.DataArray(
        data=A, coords={COORD1_LABEL: dim1_coords, COMPONENT_LABEL: rank_labels}
    )
    xr_B = xr.DataArray(
        data=B,
        coords={
            COORD1_LABEL: dim1_coords,
            COORD2_LABEL: dim2_coords,
            COMPONENT_LABEL: rank_labels,
        },
    )
    xr_C = xr.DataArray(
        data=C, coords={COORD3_LABEL: dim3_coords, COMPONENT_LABEL: rank_labels}
    )

    parafac2_ds = xr.Dataset(data_vars={A_LABEL: xr_A, BS_LABEL: xr_B, C_LABEL: xr_C})
    return parafac2_ds


def comp_slices_to_xr(parafac2_ds: xr.Dataset) -> xr.DataArray:
    # now compute the componant slices. From memory it is computed as the outer product of the three tensors?

    if not isinstance(parafac2_ds, xr.Dataset):
        raise TypeError("Expect a Dataset")

    for label in [A_LABEL, BS_LABEL, C_LABEL]:
        if label not in parafac2_ds.variables:
            raise RuntimeError(
                f"expect {A_LABEL}, {BS_LABEL}, and {C_LABEL} to be in the input dataset"
            )

    comp_slices = np.einsum(
        "ir, ijr, kr ->irjk",
        parafac2_ds[A_LABEL],
        parafac2_ds[BS_LABEL],
        parafac2_ds[C_LABEL],
    )

    coords = {dim: parafac2_ds.coords[dim] for dim in parafac2_ds.dims}
    comp_slice_xr = xr.DataArray(data=comp_slices, coords=coords)

    return comp_slice_xr


def compute_reconstruction_from_slices(components: xr.DataArray) -> xr.DataArray:
    """
    Compute the PARAFAC2 reconstruction from individual component slices. Requires the
    components to be indexed along the second mode for the sum to be correct.

    Parameters
    ==========

    components: xr.DataArray
        a 3 mode array of the component slices with the slices as the 2nd mode.
    """

    recon_data = np.einsum("irjk->ijk", components)

    dim_coords = list(components.dims)
    dim_coords.remove(COMPONENT_LABEL)

    input_coords = components.coords
    coords = {k: v for k, v in dict(input_coords).items() if k in dim_coords}

    recon_xr = xr.DataArray(data=recon_data, coords=coords)

    return recon_xr


def parafac2_pipeline(
    input_data: xr.DataArray, rank: int, parafac2_args={}
) -> xr.Dataset:
    """compute a PARAFAC2 decomposition on an xr.DataArray. returning an xr.Dataset with the model tensors, component slices and reconstruction as variables"""

    _decomp, err = parafac2(
        tensor_slices=input_data.to_numpy(), rank=rank, **parafac2_args
    )

    parafac2_ds = decomp_as_xr(
        input_data=input_data,
        rank=rank,
        decomp=_decomp,
    )

    ds = xr.merge([input_data.rename("input_data"), parafac2_ds])

    ds = ds.assign(components=comp_slices_to_xr(parafac2_ds=parafac2_ds))

    ds = ds.assign(recon=compute_reconstruction_from_slices(components=ds.components))

    return ds
