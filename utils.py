import numpy as np
import pandas as pd
from datetime import datetime

GKX_X_COLS = ["mvel1", "beta", "betasq", "chmom", "dolvol", "idiovol", "indmom",
              "mom1m", "mom6m", "mom12m", "mom36m", "pricedelay", "turn",
              "absacc", "acc", "age", "agr", "cashdebt", "cashpr", "cfp",
              "cfp_ia", "chatoia", "chcsho", "chempia", "chinv", "chpmia",
              "convind", "currat", "depr", "divi", "divo", "dy", "egr", "ep",
              "gma", "grltnoa", "herf", "hire", "invest", "lev", "lgr",
              "mve_ia", "operprof", "orgcap", "pchcapx_ia", "pchcurrat",
              "pchdepr", "pchgm_pchsale", "pchquick", "pchsale_pchinvt",
              "pchsale_pchrect", "pchsale_pchxsga", "pchsaleinv", "pctacc",
              "ps", "quick", "rd", "rd_mve", "rd_sale", "realestate", "roic",
              "salecash", "saleinv", "salerec", "secured", "securedind", "sgr",
              "sin", "sp", "tang", "tb", "cash", "chtx", "cinvest", "ear",
              "nincr", "roaq", "roavol", "roeq", "rsup", "stdacc", "stdcf",
              "ms", "baspread", "ill", "maxret", "retvol", "std_dolvol",
              "std_turn", "zerotrade", "bm", "bm_ia",]

SIC_DIVISIONS = [
    (0, 0, "Nonclassifiable"),
    (1, 9, "Agriculture, Forestry & Fishing"),
    (10, 14, "Mining"),
    (15, 17, "Construction"),
    (20, 39, "Manufacturing"),
    (40, 49, "Transportation & Utilities"),
    (50, 51, "Wholesale Trade"),
    (52, 59, "Retail Trade"),
    (60, 67, "Finance, Insurance & Real Estate"),
    (70, 89, "Services"),
    (91, 99, "Public Administration"),
]
  
  
def sic2_to_industry(sic2, unknown="Unknown"):
    """Map a 2-digit SIC code (or a Series/array of them) to its division name."""
    def one(code):
        if pd.isna(code):
            return unknown
        code = int(code)
        for lo, hi, name in SIC_DIVISIONS:
            if lo <= code <= hi:                                            
                return name
        return unknown

    if isinstance(sic2, pd.Series):
        return sic2.map(one)
    if isinstance(sic2, (list, tuple, np.ndarray)):                         
        return np.array([one(c) for c in sic2])
    return one(sic2)


def make_data(n, p, beta, noise=1.0, corr=0.0, seed=0,
              positive_coefs=False, noise_type="gaussian", toeplitz_rho=0.0):
    """Generate y = X beta + noise.

    n              : number of observations
    p              : number of features
    beta           : true regression coefs
    noise          : scale of the noise (bigger -> lower signal)
    corr           : if > 0, makes the first few features correlated (elastic net)
    positive_coefs : if True, all true coefficients are positive (for NNLS demo)
    noise_type     : "gaussian" (thin tails) or "t" (heavy tails / outliers,
                     for showing L1 vs L2 loss). "t" uses a Student-t with 2
                     degrees of freedom, which produces occasional huge errors.
    toeplitz_rho   : if > 0, draw X from a multivariate normal with an AR(1)
                     covariance Sigma[i,j] = rho^|i-j|. Features near each
                     other in index are correlated, decaying with distance.
    Returns X, y, and the true beta (we KNOW the answer because we made it).
    """
    rng = np.random.default_rng(seed)

    if toeplitz_rho > 0:
        # AR(1) / Toeplitz covariance: Sigma[i,j] = rho^|i-j|
        idx = np.arange(p)
        Sigma = toeplitz_rho ** np.abs(idx[:, None] - idx[None, :])
        X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    else:
        X = rng.standard_normal((n, p))

    # optionally make the first 3 features highly correlated
    if corr > 0:
        base = rng.standard_normal(n)
        for j in range(min(3, p)):
            X[:, j] = corr * base + (1 - corr) * rng.standard_normal(n)

    if noise_type == "t":
        # Student-t with df=2: heavy tails, occasional large outliers.
        eps = rng.standard_t(df=2, size=n)
    else:
        eps = rng.standard_normal(n)
    y = X @ beta + noise * eps
    return X, y


class FastOLS:
    def __init__(self, ridge=1e-8, dtype=np.float32):
        self.ridge = ridge
        self.dtype = dtype

    def fit(self, X, y):
        X = np.asarray(X, dtype=self.dtype)
        y = np.asarray(y, dtype=self.dtype)
        p = X.shape[1]
        XtX = X.T @ X
        Xty = X.T @ y
        if self.ridge > 0:
            XtX.flat[:: p + 1] += self.ridge
        self.coef_ = np.linalg.solve(XtX, Xty).astype(self.dtype)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=self.dtype)
        return (X @ self.coef_).astype(self.dtype)

class FastHuberRidge:
    def __init__(self, delta_q=99.9, max_iter=10, tol=1e-5):
        self.delta_q = delta_q
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y, alpha=0.0):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        n, p = X.shape
        R = np.zeros((p, p))
        R.flat[::p + 1] = alpha

        XtX = X.T @ X
        Xty = X.T @ y

        A = XtX + R
        b = Xty
        beta = np.linalg.solve(A, b)

        for _ in range(self.max_iter):
            old = beta.copy()
            r = y - X @ beta
            abs_r = np.abs(r)
            delta = np.percentile(abs_r, self.delta_q)
            down = abs_r > delta
            if np.any(down):
                w_down = delta / (abs_r[down] + 1e-12)      # < 1
                shrink = 1.0 - w_down                        # how much to remove
                X_down = X[down]
                A = XtX + R - X_down.T @ (X_down * shrink[:, None])
                b = Xty - X_down.T @ (shrink * y[down])
            else:
                A = XtX + R
                b = Xty

            beta = np.linalg.solve(A, b)
            if np.linalg.norm(beta - old) < self.tol:
                break

        self.coef_ = beta
        return self

    def predict(self, X):
        return np.asarray(X) @ self.coef_


def safe_corr(y, yhat, min_obs=3, eps=1e-12, method="pearson"):
    assert method in ("pearson", "spearman")
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[mask], yhat[mask]
    if y.size < min_obs:
        return np.nan
    if method == "spearman":
        from scipy.stats import rankdata
        y, yhat = rankdata(y), rankdata(yhat)
    y -= y.mean()
    yhat -= yhat.mean()
    denom = np.sqrt(np.sum(y**2) * np.sum(yhat**2))
    if denom < eps:
        return np.nan
    return float(np.sum(y * yhat) / denom)

def monthly_ic(df, y_col="y", pred_col="yhat", date_col="date",
               corr_method="pearson", pooled=False):
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    assert corr_method in ["pearson", "spearman"]
    if pooled:
        return df[[y_col, pred_col]].corr(method=corr_method).iloc[0, 1]
    out = (
        df.groupby(date_col)
          .apply(lambda g: g[[y_col, pred_col]].corr(method=corr_method).iloc[0, 1])
          .rename("ic")
    )
    return out

def decile_portfolios(df, y_col="y", pred_col="yhat", date_col="date", n_deciles=10):
    df = df.copy()
    df["decile"] = (
        df.groupby(date_col)[pred_col]
        .transform(
            lambda x: pd.qcut(x.rank(method="first"), n_deciles, labels=False) + 1
        )
    )
    decile_returns = (
        df.dropna(subset=["decile"])
          .groupby([date_col, "decile"])[y_col]
          .mean()
          .unstack("decile")
          .sort_index()
    )
    decile_returns.columns = [f"d_{col}" for col in decile_returns.columns]
    decile_returns["ls_ret"] = decile_returns[f"d_{str(n_deciles)}"] - decile_returns["d_1"]
    return decile_returns


def portfolio_stats(ls, periods_per_year=12, col = "ls_ret"):
    r = ls[col]
    mean_m, sd_m = r.mean(), r.std(ddof=1)
    return {
        "mean_monthly_%": 100 * mean_m,
        "mean_vol_%":     100 * sd_m,
        "ann_sharpe":     (mean_m / sd_m) * np.sqrt(periods_per_year),
        "cum_return_%":   100 * r.sum(),
        "n_months":       len(r),
    }


def gkx_summary_stats(preds):
    ic = monthly_ic(preds)
    ls = decile_portfolios(preds)
    return {
        "ic": ic,
        "ic_stats": ic.describe(),
        "portfolio_rets": ls,
        "portfolio_stats": pd.Series(portfolio_stats(ls)),
    }


def gkx_walkforward(df, fit_fn, x_cols=GKX_X_COLS, y_col="ret", scorer=None,
                    first_pred_year=2010, final_pred_year=2021, val_years=5,
                    max_train_years=100, **kwargs):

    y = df[y_col].to_numpy(dtype=np.float32)
    X = df[x_cols].fillna(0).to_numpy(dtype=np.float32)
    permno = df["permno"].to_numpy()
    ym = df["ym"].to_numpy()
    date = df["date"].to_numpy()
    year = ym // 100

    pred_blocks = []
    extra_info = {}

    for test_year in range(first_pred_year, final_pred_year + 1):

        val_end     = test_year - 1
        val_start   = val_end - val_years + 1
        train_end   = val_start - 1
        train_start = max(year.min(), train_end - max_train_years + 1)

        if train_end < train_start or val_start > val_end:
            print(f"{datetime.now():%H:%M:%S} | test={test_year} | "
                  f"SKIP (insufficient history: "
                  f"train={train_start}-{train_end}, val={val_start}-{val_end})")
            continue

        train_idx = (year >= train_start) & (year <= train_end)
        val_idx   = (year >= val_start)   & (year <= val_end)
        test_idx  = year == test_year
        tv_idx    = train_idx | val_idx

        X_tr, y_tr   = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_te, y_te   = X[test_idx], y[test_idx]
        X_tv, y_tv   = X[tv_idx], y[tv_idx]

        test_pred, extra_info[test_year] = fit_fn(
            X_tr, y_tr, X_val, y_val, X_tv, y_tv, X_te, y_te,
            val_ym=ym[val_idx], scorer=scorer, **kwargs
        )

        pred_blocks.append(pd.DataFrame({
            "permno": permno[test_idx],
            "date": date[test_idx],
            "ym": ym[test_idx],
            "y": y_te,
            "yhat": test_pred,
        }))

        print(
            f"{datetime.now():%H:%M:%S} | "
            f"test={test_year} | "
            f"train={train_start}-{train_end} | "
            f"val={val_start}-{val_end} | "
            f"n_train={train_idx.sum():>8,} n_test={test_idx.sum():>6,}"
        )

    preds = pd.concat(pred_blocks).reset_index(drop=True)
    return preds, extra_info
