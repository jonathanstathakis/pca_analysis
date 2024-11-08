import numpy as np
from scipy.sparse import csc_matrix


def Aptera(X, Fmin, Fmax, maxiter):
    # Input parameters
    init, ALG, Maxiters, display, PARFOR_FLAG, tol = 2, 1, 5000, 0, 0, 1e-7
    Options = [tol, Maxiters, init, 0, display, ALG, PARFOR_FLAG]

    allF = np.arange(Fmin, Fmax + 1)
    all_c = np.zeros((len(allF), maxiter))
    all_v = np.zeros((len(allF), maxiter))
    all_itr = np.zeros((len(allF), maxiter))
    lcurve = [[None for _ in range(maxiter)] for _ in range(len(allF))]

    for f_idx, f in enumerate(allF):
        for it in range(maxiter):
            print(f"Iteration {it + 1}, Running rank {f}")

            # Placeholders for the parafac2 function; customize with actual function
            B, C, P, itrNum = parafac2(X, f, [0, 0], Options)

            # Custom update function Aptera_update
            c, v, l = Aptera_update(X, B, C, P, f, Fmin, Fmax)

            all_c[f_idx, it] = c
            all_v[f_idx, it] = v
            all_itr[f_idx, it] = itrNum
            lcurve[f_idx][it] = l

    # Post-process to get F_est and Fac results
    F_est_itr = np.argmin(all_itr, axis=0)
    F_est = min(np.median(all_c, axis=1).astype(int), F_est_itr + 1)

    # Re-run parafac2 at estimated rank for final factors
    B, C, _, U = parafac2(X, F_est, [0, 0], Options)
    Fac = [U, B, C]

    return F_est, Fac


def Aptera_update(X, B, C, P, F, Fmin, Fmax):
    K, J = C.shape[0], B.shape[0]
    YY = [csc_matrix(P[k].T @ X[k]) for k in range(K)]

    # Convert to tensor form
    Y = np.array(YY).reshape((F, K, J)).transpose(1, 2, 0)

    # L-curve computation with custom function
    Rest, valest, lcurve = Aptera_lcorner(Y, Fmin, Fmax)
    return Rest, valest, lcurve


def Aptera_lcorner(T, Fmin, Fmax):
    size_tens = T.shape
    N = 3
    in_dims = [np.arange(1, i + 1) for i in size_tens]

    outval = np.array(np.meshgrid(*in_dims)).T.reshape(-1, N)

    # Placeholder for hosvd_parafac2; use actual function as needed
    _, _, sv = hosvd_parafac2(T)

    # Adjust singular values for each mode
    for n in range(N):
        z = np.zeros(size_tens[n])
        z[: len(sv[n])] = sv[n]
        sv[n] = z

    loss = [np.cumsum(s[::-1] ** 2)[::-1] for s in sv]

    # Compute relative error
    relerr = np.zeros(len(outval))
    for n, l in enumerate(loss):
        relerr += l[size_tens[n] - outval[:, n]]

    relerr = np.minimum(np.sqrt(relerr) / np.linalg.norm(sv[0]), 1)

    # L-curve culling and corner detection
    x = outval.sum(axis=1) / np.prod(size_tens)
    y = relerr
    d = np.sqrt(x**2 + y**2)
    sorted_idx = np.argsort(d)
    outval, x, y = outval[sorted_idx], x[sorted_idx], y[sorted_idx]

    idx = (x >= x[0]) & (y > y[0])
    p = 1
    while p < len(x):
        idx = (x >= x[p] & (y > y[p])) | ((x > x[p]) & (y >= y[p]))
        outval, x, y, d = outval[~idx], x[~idx], y[~idx], d[~idx]
        p += 1 - np.sum(idx[:p])

    # Estimate L-curve rank based on culling results
    size_core = outval[0]
    rankest = min(size_core[1], size_core[2])
    valest = d[0]

    return rankest, valest, [x, y, outval]
