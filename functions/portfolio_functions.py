### portfolio_functions.py

# Functions for calculating the portfolio's weights at each date


import numpy as np
import cvxpy as cp
import pandas as pd

# Reconstruct covariance matrix for portfolio construction
def reconstruct_cov_shrunk(pca_result):
    eigvecs = pca_result["eigvecs"].astype(np.float64)
    eigvals = pca_result["eigvals"].astype(np.float64)
    sigma   = pca_result["sigma"].astype(np.float64)

    rho = (eigvecs * eigvals) @ eigvecs.T
    rho = (rho + rho.T) / 2

    return rho, sigma


# Equal weight portfolio
def equal_weight_portfolio(permnos):
    n = len(permnos)
    return {p: 1.0 / n for p in permnos}


# Market-cap portfolio
def mcap_weight_portfolio(permnos, mcap_wide, t):
    mcap_t = mcap_wide.loc[t, permnos]
    w = mcap_t / mcap_t.sum()
    return w.to_dict()


# Minimum variance portofolio (long-only)
def min_variance_weights_qp(Sigma, permnos, solver=cp.OSQP):
    N = Sigma.shape[0]
    w = cp.Variable(N)
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma))),
        [cp.sum(w) == 1, w >= 0]
    )
    prob.solve(solver=solver)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    weights = np.clip(w.value, 0, None)
    weights /= weights.sum()
    return dict(zip(permnos, weights))


# Max diversification portfolio (long-only)
def max_diversification_weights_qp(rho, sigma, permnos, solver=cp.OSQP):
    N = rho.shape[0]
    u = cp.Variable(N)
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(u, cp.psd_wrap(rho))),
        [cp.sum(u) == 1, u >= 0]
    )
    prob.solve(solver=solver)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    u_val = np.clip(u.value, 0, None)
    w = u_val / sigma
    w /= w.sum()
    return dict(zip(permnos, w))


# Equal-risk contribution portfolio
def erc_weights_newton(Sigma, permnos, tol=1e-7, max_iter=50):
    N = Sigma.shape[0]
    w = np.full(N, 1.0 / N)   # equal-weight start: positive, feasible

    converged = False
    for _ in range(max_iter):
        Sw = Sigma @ w
        grad = Sw - 1.0 / w
        if np.linalg.norm(grad) < tol:
            converged = True
            break

        Hess = Sigma + np.diag(1.0 / w**2)
        d = np.linalg.solve(Hess, -grad)

        t = 1.0
        while np.any(w + t * d <= 0):        # stay in the positive orthant
            t *= 0.5

        f0 = 0.5 * w @ Sigma @ w - np.sum(np.log(w))
        slope = grad @ d
        while True:
            w_new = w + t * d
            f_new = 0.5 * w_new @ Sigma @ w_new - np.sum(np.log(w_new))
            if f_new <= f0 + 1e-4 * t * slope or t < 1e-12:
                break
            t *= 0.5

        w = w + t * d

    w = np.clip(w, 0, None)
    w /= w.sum()
    return dict(zip(permnos, w)), converged



#### Helper functions

def portfolio_weights_wide(portfolios, portfolio_name):
    records = []
    for t, port_t in portfolios.items():
        w = port_t.get(portfolio_name)
        if w is None:
            continue
        for permno, weight in w.items():
            records.append((t, permno, weight))

    df = pd.DataFrame(records, columns=["date", "permno", "weight"])
    wide = df.pivot(index="date", columns="permno", values="weight").fillna(0.0)
    return wide.sort_index()


def smoothed_portfolio_dict(portfolios, portfolio_name, window=20):
    wide = portfolio_weights_wide(portfolios, portfolio_name)
    smoothed_wide = wide.rolling(window=window, min_periods=window).mean()
    smoothed_wide = smoothed_wide.dropna(how="all")   # drop first (window-1) dates, no full window yet

    smoothed = {}
    for t, row in smoothed_wide.iterrows():
        nz = row[row > 1e-10]     # drop numerically-zero entries, keep dict small
        smoothed[t] = nz.to_dict()
    return smoothed