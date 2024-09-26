---
cdt: 2024-09-03T12:30:17
description: "a summary of the process to reconstruct the tensor from the PARAFAC2 decomposition as implemented by Tensorly"
title: "How TensorLy's PARAFAC2 `parafac_to_tensor` Works"
---

One method of identifying which is the scaling matrix is to observe the tensorly `parafac2_to_tensor` code, which is a function that reconstructs the 3-way tensor from the decomposition factors. Following is a sumamary of its contents:

1. the A, C, and projections are unpacked from the `Parafac2_Tensor` object
2. the slices are constructed from the `parafac2_to_slices` function
3. The number of rows of each projection matrix is calculated (equal to $J$).
4. The tensor is created by:
    1. creating a three way numpy array of zeros with shape equal to $I$, $J$, $K$ and same datatype
    2. iterating over the slices and updating each slice of the zero tensor created in (1) with a slice from the slices array.
5. return

Which begs the question, how does `parafac2_to_slices` work?

1. validate `parafac2_tensor` input
2. unpack `parafac2_tensor` object
3. Multiply A by the weights, set weights to None
4. pack the unpacked objects into a tuple and nested tuple the same as the object unpacked in (2)
5. get the number of rows from A, equal to $I$
6. iterate over 1 to $I$ calling `parafac2_to_slice` on the tuple created in (5), and providing the index of the current iteration over $I$.
7. return the list of slices created in (6)

So how does `parafac2_to_slice` work?

1. Takes a `parafac2_tensor` (in this context a tuple with three elements (see `parafac2_to_slices` (4)), a slice index (from 1 to k)
2. unpack `parafac2_tensor` into `weights`, `A`, `B`, `C`, and `projections`
3. initialise an `a` as the `slice_idx`th slice of `A`. It is an array of shape $1 x r$
4. multiply `a` by the weights. Note that in the call from `parafac2_to_slices`, $A$ is scaled by the weights prior to the call to `parafac2_to_slice` and the weights are set to None, meaning that this step is skipped.
5. initialise `Ct` as the transpose of `C`
6. initialise `B_i` as the dot product of the `slice_idx`th projection matrix and `B`
7. return the dot product of `B_i` * a and `Ct`, producing the `slice_idx`th slice of the tensor.

So thats it.

In summary, the algorithm to reconstruct the tensor is as follows, starting from the decomposition object

1. For each $i$ in $I$, calculate: $X_i = P_i B a_i C ^ \top$
2. arrange as a three way numpy array by iterating over the $i$

Quite simple, and as an aside we can see how the rows of $A$ are used to scale $B_i$.