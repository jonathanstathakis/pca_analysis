import numpy as np
from numpy.linalg import norm
from scipy.linalg import orth


def create_parafac2_problem(Imax, J, K, F, Fn, noisePercent, PARFOR_FLAG=False):
    a = np.ceil(Imax / 2).astype(int)
    b = Imax

    comm = round(K / F)
    Crange = np.arange(1, K + 1, comm)
    Crange = np.append(Crange, K)

    # Initialize C with zeros, then set block structure
    C = np.zeros((K, F))

    for i in range(0, F):
        stIdx = Crange[i]
        edIdx = Crange[i + 1] - 1
        C[stIdx:edIdx, i] = 1

    # TODO: continue translating the matlab code to Python..
    # Random matrices
    C = np.random.rand(K, F)
    C_n = np.random.rand(K, Fn)
    B = np.random.rand(J, F)
    B_n = np.random.rand(J, Fn)

    # Orthogonal matrices H and H_n
    H = orth(np.random.rand(F).T)
    H_n = orth(np.random.rand(Fn).T)

    print(H.shape, H_n.shape)
    # Create P and P_n matrices

    P = []
    P_n = []

    if PARFOR_FLAG:
        ...
    else:
        for i in range(1, K + 1):
            r = (b - a) * np.random.rand() + a
            P.append(orth(np.random.rand(r, F)))
            P_n.append(orth(np.random.rand(r, Fn)))

    # Generate X and noise matrices
    X = []
    noise = []
    totalnnz = 0
    for i in range(K):
        # Core tensor construction
        X_i = (B @ np.diag(C[i, :]) @ (P[i] @ H).T).T
        X.append(X_i)

        # Adding structured noise
        noise_i = (B_n @ np.diag(C_n[i, :]) @ (P_n[i] @ H_n).T).T
        norm_data = norm(X_i)
        norm_noise = norm(noise_i)
        scaled_noise = noise_i * (1 / norm_noise) * (norm_data * noisePercent / 100)
        X[i] += scaled_noise  # Add noise to X_i
        noise.append(scaled_noise)

        totalnnz += np.count_nonzero(X_i)

    return X, totalnnz, noise


if __name__ == "__main__":
    Fo = 5
    Fmin = 2
    Fmax = 2 * Fo
    Fn = 50
    noise_percent = 20
    max_iter = 3
    Imax = 200
    J = Imax + 10
    K = Imax + 20
    PARFOR_FLAG = False

    result = create_parafac2_problem(Imax, J, K, Fo, Fn, noise_percent, PARFOR_FLAG)

    print(result)
