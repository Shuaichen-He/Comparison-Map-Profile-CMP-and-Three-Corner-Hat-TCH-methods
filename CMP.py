import xarray as xr
import numpy as np
from numba import njit, prange
import pandas as pd
import warnings
from scipy.stats import pearsonr
from itertools import combinations

warnings.simplefilter("ignore", category=RuntimeWarning)


# Define a function to calculate the standard deviation (sigma) of a 2D array.
@njit(parallel=True)
def sigma(data, L, W):
    """
    Calculate the standard deviation (sigma) of a 2D array while ignoring NaN values.

    Parameters:
    - data: 2D array of data values.
    - L: Number of rows in the array.
    - W: Number of columns in the array.

    Returns:
    - sigma: Standard deviation of the data.
    """
    mean_value = np.nanmean(data)
    SS = 0  # Sum of squared differences

    for l in prange(L):
        for w in prange(W):
            if np.isnan(data[l, w]):
                ss = 0  # Ignore NaN values
            else:
                ss = (data[l, w] - mean_value) ** 2
            SS += ss

    # Calculate the standard deviation
    sigma = np.sqrt(SS / ((L * W) - 1))
    return sigma


# Define a function to calculate the correlation coefficient (CC) between two 2D arrays.
def CC(d1, d2, L, W, sigma2):
    """
    Calculate the cross-correlation (CC) between two 2D arrays.

    Parameters:
    - d1, d2: 2D arrays of data values.
    - L: Number of rows in the arrays.
    - W: Number of columns in the arrays.
    - sigma2: squared sigma of d1 and d2.

    Returns:
    - CC: cross-Correlation.
    """
    C = 0
    mean_d1 = np.nanmean(d1)
    mean_d2 = np.nanmean(d2)

    for l in prange(L):
        for w in prange(W):
            if not (np.isnan(d1[l, w]) or np.isnan(d2[l, w])):
                cc = (d1[l, w] - mean_d1) * (d2[l, w] - mean_d2) / sigma2
            else:
                cc = 0
            C += cc

    CC = C / (L * W)
    return CC


# Define a function to calculate CMP distance.
def CMP_distance(r, x, y, data1, data2, mask):
    """
    Calculate the CMP distance between two datasets.

    Parameters:
    - r: Maximum radius for comparison.
    - x, y: Dimensions of the data arrays.
    - data1, data2: 2D arrays of data values.
    - mask: Spatial mask for valid regions.

    Returns:
    - s: Distance matrix for each radius.(The abs is calculated directly between the two images at position 0)
    - multi_mean_pic: Mean distance picture across all radii.
    - multi_mean: Global mean distance.
    - mean_distance: Mean distance for each radius.
    - std_distance: Standard deviation of distances for each radius.
    """
    s = np.zeros((r + 1, x, y))  # Initialize the distance matrix

    for R in prange(r + 1):
        for i in prange(x):
            for j in prange(y):
                # Define the local window
                a, b = max(0, i - R), min(x - 1, i + R)
                c, d = max(0, j - R), min(y - 1, j + R)

                # Calculate absolute mean differences within the local window
                s[R, i, j] = abs(
                    np.nanmean(data1[a : b + 1, c : d + 1])
                    - np.nanmean(data2[a : b + 1, c : d + 1])
                )

        # Apply the spatial mask
        s[R, :, :] *= mask

    multi_mean_pic = np.nanmean(s[1:], axis=0)
    multi_mean = np.nanmean(s[1:])
    mean_distance = np.nanmean(s, axis=(1, 2))
    std_distance = np.nanstd(s, axis=(1, 2))

    return s, multi_mean_pic, multi_mean, mean_distance, std_distance


# Define a function to calculate CMP correlation coefficient (CC).
def CMP_CC(r1, r, x, y, data1, data2, mask):
    """
    Calculate the CMP cross-correlation (CC) for a given radius.

    Parameters:
    - r1, r: Range of radii for comparison.
    - x, y: Dimensions of the data arrays.
    - data1, data2: 2D arrays of data values.
    - mask: Spatial mask for valid regions.

    Returns:
    - s: CC matrix for each radius. (The cross-correlation is calculated directly between the two images at position 0)
    - multi_mean_pic: Mean CC picture across all radii.
    - multi_mean: Global mean CC.
    - mean_CC: Mean CC for each radius.
    - std_CC: Standard deviation of CCs for each radius.
    """
    s = np.zeros((r + 1, x, y))

    # Global CC for the entire dataset
    s[0, 0, 0] = Global_CC(data1, data2, x, y)

    for R in prange(r1, r + 1):
        for i in prange(x):
            for j in prange(y):
                # Define the local window
                a, b = max(0, i - R), min(x - 1, i + R)
                c, d = max(0, j - R), min(y - 1, j + R)

                # Extract local windows and calculate CC
                d1 = data1[a : b + 1, c : d + 1]
                d2 = data2[a : b + 1, c : d + 1]
                L, W = d1.shape
                sigma2 = sigma(d1, L, W) * sigma(d2, L, W)
                s[R, i, j] = CC(d1, d2, L, W, sigma2)

        # Apply the spatial mask
        s[R, :, :] *= mask

    multi_mean_pic = np.nanmean(s[r1 + 1 :], axis=0)
    multi_mean = np.nanmean(s[r1 + 1 :])
    mean_CC = np.nanmean(s[r1 + 1 :], axis=(1, 2))
    std_CC = np.nanstd(s[r1 + 1 :], axis=(1, 2))

    return s, multi_mean_pic, multi_mean, mean_CC, std_CC


# Define a function for the global CC calculation.
def Global_CC(data1, data2, L, W):
    """
    Calculate the global cross-correlation (CC) for the entire dataset.

    Parameters:
    - data1, data2: 2D arrays of data values.
    - L, W: Dimensions of the data arrays.

    Returns:
    - result: Global CC.
    """
    result = CC(data1, data2, L, W, sigma(data1, L, W) * sigma(data2, L, W))
    return result


# Main execution block
if __name__ == "__main__":
    mera = np.load("./mera.npy", allow_pickle=True)
    mflux = np.load("./mflux.npy", allow_pickle=True)
    mgldas = np.load("./mgldas.npy", allow_pickle=True)
    mgleam = np.load("./mgleam.npy", allow_pickle=True)
    # Load the spatial mask
    mask = xr.open_dataset(
        "./mask.nc"
    ).Band1.values

    # Define combinations of datasets
    combins = [c for c in combinations([mera, mflux, mgldas, mgleam], 2)]
    print([c for c in combinations(["ERA5", "FluxCom", "GLDAS", "GLEAM"], 2)])

    # Calculate Mean_CC for each combination
    # Input parameters should account for scope variations (1-15) and image dimensions (62×46)
    Mean_CC = []
    for i in range(6):
        %time Mean_CC.append(CMP_CC(1, 15, 62, 46, combins[i][0], combins[i][1], mask))

    # Save the results to a file
    np.save(
        "./Mean_CC.npy",
        pd.Series(Mean_CC),
        allow_pickle=True,
    )
