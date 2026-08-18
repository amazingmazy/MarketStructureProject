### diversifications_functions.py

# Functions for calculating the diversification measures

import numpy as np

### Calculate the fraction of portfolio variance explained by each component
# Important: weights must be a dictionary {permno: weight}
def get_portfolio_variance_shares(weights, pca_result):

    permnos = pca_result["permnos"]
    eigvecs = pca_result["eigvecs"].astype(np.float64)
    eigvals = pca_result["eigvals"].astype(np.float64)
    sigma   = pca_result["sigma"].astype(np.float64)

    # Align the weights so that they match the eigenvectors stock order
    w = np.array([weights.get(p, 0.0) for p in permnos])   

    # Scale the portfolio weight by the volatility
    u = w * sigma      

    # Project the scaled weights onto each component
    scores = eigvecs.T @ u                                  

    # Contribution of variance of each component
    var_contrib = eigvals * scores**2
    total_var = var_contrib.sum()

    if total_var <= 0:
        return None, 0.0

    # Fraction of variance explained by each component
    p_k = var_contrib / total_var

    return p_k, total_var


# Entropy-based measure of diversification
def n_entropy(p_k):
    nonzero = p_k > 0                      # avoid 0 * log(0)
    entropy = -np.sum(p_k[nonzero] * np.log(p_k[nonzero]))
    return np.exp(entropy)

# HHI-based measure of diversification
def n_hhi(p_k):
    return 1.0 / np.sum(p_k**2)

# Variance explained by top N components measure
def top_n_var_explained(p_k, n=10):
    return np.sum(p_k[:n])

# Wrapper function
def compute_diversification_measures(weights, pca_result, top_n=10):

    p_k, total_var = get_portfolio_variance_shares(weights, pca_result)
    
    if p_k is None:
        return None

    return {
        "portfolio_variance": total_var,
        "p_k": p_k,
        "enb_entropy": n_entropy(p_k),
        "enb_hhi": n_hhi(p_k),
        f"top_{top_n}_var_explained": top_n_var_explained(p_k, top_n),
    }