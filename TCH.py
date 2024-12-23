import numpy as np
import pandas as pd
import xarray as xr
from scipy import optimize


def TCH(arr, G_T):
    """
    Perform TCH analysis on a given array.

    Parameters:
    - arr: numpy.ndarray
        Input 2D array where rows are samples and columns are features.
    - G_T: bool
        There are two constraints that can be used to solve F
        Determines whether to use the G (True) or H (False).

    Returns:
    - tch: numpy.ndarray
        The diagonal elements of the optimized matrix R (target correlation values).
    - uct: numpy.ndarray
        The uncertainty values for each feature.
    - r_uct: numpy.ndarray
        The relative uncertainty for each feature.
    """

    def Fun(r):
        """
        Objective function with normalization factor.
        """
        f = 0
        for j in range(len(S)):
            f += np.power(r[j], 2)
            for k in range(len(S)):
                if j < k:
                    f += np.power(S[j, k] - r[len(S)] + r[j] + r[k], 2)
        K = np.linalg.det(S).item()  # Convert determinant to scalar
        F = f / np.power(K, 2 / len(S))  # Normalized objective function
        return F

    def fun(r):
        """
        Simplified objective function (without normalization).
        """
        f = 0
        for j in range(len(S)):
            f += np.power(r[j], 2)
            for k in range(len(S)):
                if j < k:
                    f += np.power(S[j, k] - r[len(S)] + r[j] + r[k], 2)
        return f

    # Dimensions of the input array
    M, N = np.shape(arr)

    # Reference column (last column)
    ref_arr = arr[:, N - 1]

    # Target columns (all columns except the last one)
    tar_arr = arr[:, 0 : N - 1]

    # Difference matrix Y (used to calculate covariance)
    Y = tar_arr - np.repeat(np.reshape(ref_arr, (M, 1)), N - 1, axis=1)

    # Covariance matrix of Y (size: (N-1) x (N-1))
    S = np.cov(Y.T)

    # Initialize an empty matrix R
    Q = np.zeros((N, N))

    # Row vector of ones
    u = np.ones((1, N - 1))

    # Initialize the bottom-right element of R
    Q[N - 1, N - 1] = 1 / (2 * np.dot(np.dot(u, np.linalg.inv(S)), u.T).item())

    # Initial values for optimization
    x0 = Q[:, N - 1]

    # Define constants for constraints
    u_ = np.ones((1, len(S)))
    det_S = np.linalg.det(S).item()  # Convert determinant to scalar
    inv_S = np.linalg.inv(S)  # Inverse of S
    K1 = np.power(det_S, 1 / len(S))  # K = (|S|)^(1/len(S))

    # H Constraint conditions
    H = {
        "type": "ineq",
        "fun": lambda r: -1
        * (
            r[len(S)]
            - np.dot(
                np.dot(np.reshape(r[0:-1], (1, len(r) - 1)) - r[len(S)] * u_, inv_S),
                (np.reshape(r[0:-1], (1, len(r) - 1)) - r[len(S)] * u_).T,
            )
        ).item()
        / K1,
    }

    # G Constraint conditions
    G = {"type": "ineq", "fun": lambda r: fun(r) / N}

    # Perform optimization
    if G_T:
        x = optimize.minimize(Fun, x0, method="COBYLA", tol=2e-10, constraints=G)
    else:
        x = optimize.minimize(Fun, x0, method="COBYLA", tol=2e-10, constraints=H)

    # Update the last column of R with the optimized values
    Q[:, N - 1] = x.x

    # Fill in the upper triangle of R
    for ii in range(N - 1):
        for jj in range(ii, N - 1):
            Q[ii, jj] = S[ii, jj] - Q[N - 1, N - 1] + Q[ii, N - 1] + Q[jj, N - 1]

    # Reflect to fill the lower triangle
    Q += Q.T - np.diag(Q.diagonal())

    # Ensure symmetry of R
    for ii in range(N):
        for jj in range(ii, N):
            Q[jj, ii] = Q[ii, jj]

    # Extract results
    tch = Q.diagonal()  # Diagonal elements of R (target correlation)
    uct = np.sqrt(Q.diagonal())  # Uncertainty values
    r_uct = uct / np.mean(abs(arr), axis=0)  # Relative uncertainty

    return tch, uct, r_uct


# This is an example of using TCH
# You need to pre-load four datasets (ERA5, FLUXCOM, GLDAS, and GLEAM are used as examples)
# Assume each dataset is represented as a 62x46 two-dimensional grid
if __name__ == "__main__":
    era = np.load("./era.npy", allow_pickle=True)
    flux = np.load("./flux.npy", allow_pickle=True)
    gldas = np.load("./gldas.npy", allow_pickle=True)
    gleam = np.load("./gleam.npy", allow_pickle=True)

    a_h = np.zeros((62, 46))
    f_h = np.zeros((62, 46))
    s_h = np.zeros((62, 46))
    m_h = np.zeros((62, 46))

    for i in range(62):
        for j in range(46):
            arr = (
                np.array(
                    [
                        era[:, i, j].values,
                        flux[:, i, j].values,
                        gldas[:, i, j].values,
                        gleam[:, i, j].values,
                    ]
                )
                .reshape(4, 408)
                .T
            )
            t = TCH(arr, False)
            a_h[i, j] = t[2][0]
            f_h[i, j] = t[2][1]
            s_h[i, j] = t[2][2]
            m_h[i, j] = t[2][3]

    print(
        f"The regional average relative uncertainty is:\n{np.nanmean(a_h)}\n{np.nanmean(f_h)}\n{np.nanmean(s_h)}\n{np.nanmean(m_h)}"
    )
