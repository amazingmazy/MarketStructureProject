### pca_functions.py

# Functions needed for PCA, including ewma covariance, shrinkage

import numpy as np


# Exponentially-weighted covariance matrix
def ewma_cov(X, halflife):
    T = X.shape[0]
    beta = np.exp(np.log(0.5) / halflife)
    lags = np.arange(T - 1, -1, -1)      # newest row -> lag 0
    w = beta ** lags
    w /= w.sum()

    wmean = (w[:, None] * X).sum(axis=0)
    Xc = X - wmean
    cov = (Xc * w[:, None]).T @ Xc

    cov /= (1 - np.sum(w**2))            # bias correction (sample covariance matrix)

    return cov


# Exponentially-weighted mean and std for standarization of returns
def ewma_mean_std(X, halflife):
    T = X.shape[0]
    beta = np.exp(np.log(0.5) / halflife)
    lags = np.arange(T - 1, -1, -1)
    w = beta ** lags
    w /= w.sum()

    mean = (w[:, None] * X).sum(axis=0)
    var = (w[:, None] * (X - mean)**2).sum(axis=0) / (1 - np.sum(w**2))
    return mean, np.sqrt(var)


# Shrink correlation matrix
def shrink_cov_matrix(C, delta, target="constant_correlation"):
    N = C.shape[0]

    # Zero correlation target
    if target == "identity":
        F = np.eye(N)

    # Constant correlation target
    elif target == "constant_correlation":
        off_diag_sum = C.sum() - np.trace(C)
        avg_corr = off_diag_sum / (N * (N - 1))
        F = np.full((N, N), avg_corr)
        np.fill_diagonal(F, 1.0)

    else:
        raise ValueError("target must be 'identity' or 'constant_correlation'")

    return delta * F + (1 - delta) * C



######## Shrink factor

def _shrinkage_delta_from_weights(X, w, target="constant_correlation"):
    T, N = X.shape
    s = np.sum(w ** 2)
    T_eff = 1.0 / s
    q = 1.0 - s

    mean = (w[:, None] * X).sum(axis=0)
    var = (w[:, None] * (X - mean) ** 2).sum(axis=0) / q
    std = np.sqrt(var)
    Z = (X - mean) / std
    Zc = Z - (w[:, None] * Z).sum(axis=0)

    S_raw = (Zc * w[:, None]).T @ Zc
    C = S_raw / q

    y = Zc ** 2
    pi_mat = (y * w[:, None]).T @ y - S_raw ** 2
    pi_hat = pi_mat.sum()

    if target == "identity":
        F = np.eye(N)
        rho_hat = np.trace(pi_mat)
    elif target == "constant_correlation":
        off_diag_sum = C.sum() - np.trace(C)
        r_bar = off_diag_sum / (N * (N - 1))
        F = np.full((N, N), r_bar)
        np.fill_diagonal(F, 1.0)

        term1 = (w[:, None] * (Zc ** 3)).T @ Zc
        theta_mat = term1 - q * S_raw
        np.fill_diagonal(theta_mat, 0.0)
        rho_hat = np.trace(pi_mat) + r_bar * theta_mat.sum()
    else:
        raise ValueError("target must be 'identity' or 'constant_correlation'")

    gamma_hat = np.sum((F - C) ** 2)
    if gamma_hat < 1e-16:
        return 0.0

    kappa_hat = (pi_hat - rho_hat) / gamma_hat
    return float(np.clip(kappa_hat / (T_eff * q ** 2), 0.0, 1.0))


def ewma_shrinkage_delta(X, halflife, target="constant_correlation"):
    """
    X        : (T, N) return matrix, oldest -> newest
    halflife : EWMA half-life in days (e.g. 252)
    target   : "identity" or "constant_correlation"

    Returns delta in [0, 1] -- the EWMA-consistent Ledoit-Wolf shrinkage intensity.
    """
    T = X.shape[0]
    beta = np.exp(np.log(0.5) / halflife)
    w = beta ** np.arange(T - 1, -1, -1)
    w /= w.sum()
    return _shrinkage_delta_from_weights(X, w, target)