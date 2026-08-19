# Code review — `functions/portfolio_functions.py`

**Reviewer:** Bangjie · **Requested by:** Enrique

**Verdict: the five portfolio constructors and the helper functions are mathematically
correct.** The QP reductions, the ERC log-barrier Newton step, and the weight-smoothing budget
preservation all check out against the standard references. Nothing below is a bug — the items are
robustness / clarity improvements you can take or leave before the full run.

---

## What is correct as written

- **`reconstruct_cov_shrunk`** — `(eigvecs * eigvals) @ eigvecs.T` correctly rebuilds `V Λ Vᵀ`
  (each column `v_j` scaled by `λ_j`), and the symmetrisation is a sound numerical guard. Returns the
  **correlation** matrix `ρ` plus `σ`, which the caller combines as `Σ = outer(σ,σ) * ρ`. Correct.

- **`min_variance_weights_qp`** — textbook long-only min-variance QP:
  `min wᵀΣw s.t. Σw = 1, w ≥ 0`, with `psd_wrap` to tolerate a non-PSD `Σ`. The `clip(0)` +
  renormalise cleanly handles OSQP's tiny negative outputs.

- **`max_diversification_weights_qp`** — the substitution `u = D·w` (so `uᵢ = σᵢ wᵢ`) turns the
  diversification-ratio maximisation `max (wᵀσ)/√(wᵀΣw)` into `min uᵀC u s.t. Σu = 1, u ≥ 0`, then
  recovers `wᵢ = uᵢ/σᵢ` and renormalises. This is exactly the Choueifaty–Coignard reduction. Correct.

- **`erc_weights_newton`** — minimises `½ wᵀΣw − Σ ln wᵢ`. Gradient `Σw − 1/w` and Hessian
  `Σ + diag(1/w²)` are exact; the barrier + positivity backtracking keep iterates feasible; the
  normalised solution is scale-invariant, so the equal-risk-contribution property holds. Correct.

- **`smoothed_portfolio_dict` / `portfolio_weights_wide`** — because every daily weight row sums to 1
  after `fillna(0)`, the 20-day rolling **mean** of those rows also sums to 1, so smoothing preserves
  the budget constraint even when universe membership changes. `min_periods=window` ⇒ **no look-ahead**.

---

## Suggested improvements (robustness / clarity)

1. **Naming + unit hygiene in `reconstruct_cov_shrunk`.** It returns a *correlation* matrix, not a
   covariance — consider renaming to `reconstruct_corr_shrunk`. After the `float32` round-trip and
   eigenvalue clipping, the reconstructed diagonal can drift slightly from 1.0. Renormalise so implied
   vols stay exactly `σ_i`:
   ```python
   d = np.sqrt(np.clip(np.diag(rho), 1e-12, None))
   rho = rho / np.outer(d, d)
   ```

2. **`mcap_weight_portfolio` uses contemporaneous mcap at `t`.** Fine for measuring the *current*
   allocation's diversification, but the index elsewhere uses *lagged* mcap — worth a one-line comment
   so the choice is explicit. Add a guard against NaN / zero-sum:
   ```python
   mcap_t = mcap_wide.loc[t, permnos].dropna()
   if mcap_t.sum() <= 0:
       return None
   ```

3. **Solver robustness.** OSQP can be inaccurate on 500-dim QPs; accepting `optimal_inaccurate` + clip
   is pragmatic but slightly biases weights. Consider `cp.CLARABEL`, or OSQP with `polish=True` and
   tighter `eps_abs`/`eps_rel`, and log how often each solver returns the inaccurate flag.

4. **Failure handling leaves gaps.** `min_variance_weights_qp` / `max_diversification_weights_qp`
   return `None` on failure, which drops that date from the diversification panel. A warm-start from the
   previous day's weights (or an equal-weight fallback) would keep the time series continuous.

5. **Use the ERC convergence flag.** `erc_weights_newton` returns `converged`, but the build loop only
   counts it. If `erc_not_converged` is non-trivial, raise `max_iter` or loosen `tol`, and consider
   excluding non-converged dates from the analysis.

**Net:** no correctness changes required before running the full analysis; the above are polish.
