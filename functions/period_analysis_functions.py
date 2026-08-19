### period_analysis_functions.py
#
# Reusable tools for the "specific market period" deep-dives (Section 5 of the report).
# Everything here consumes the objects already built in project_notebook.ipynb:
#
#   panel         : tidy DataFrame from build_diversification_panel(...)
#                   columns = date, portfolio, portfolio_variance,
#                             enb_entropy, enb_hhi, top_10_var_explained, absorption_ratio
#   pca_results   : dict {date -> {permnos, eigvals, eigvecs, sigma,
#                                  explained_var_ratio, delta, delta_raw}}
#   portfolios    : dict {date -> {name -> {permno: weight}}}
#
# The functions are portfolio-agnostic and period-agnostic, so Anthony / Bangjie / Chirag
# can reuse the exact same code for GFC, the euro-zone crisis, 2017-2018, COVID, etc.
# by only changing the (start, end) window passed in.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Order and display labels kept consistent with the rest of the notebook.
PORTFOLIO_NAMES = ["equal_weight", "mcap", "min_var", "erc", "max_div"]
PORTFOLIO_LABELS = {
    "equal_weight": "Equal weight",
    "mcap": "Cap weight",
    "min_var": "Min variance",
    "erc": "Equal-risk contribution",
    "max_div": "Max diversification",
}
MEASURE_COLS = ["enb_entropy", "enb_hhi", "top_10_var_explained"]
MEASURE_LABELS = {
    "enb_entropy": "ENB (entropy)",
    "enb_hhi": "ENB (HHI)",
    "top_10_var_explained": "Top-10 variance explained",
}

# Canonical study windows. Widen/narrow as needed; every downstream helper just reads this.
# (Each teammate owns a subset of these.)
PERIODS = {
    "GFC":            ("2007-07-01", "2009-06-30"),
    "Euro-zone":      ("2010-01-01", "2012-12-31"),
    "2017-2018":      ("2017-01-01", "2018-12-31"),
    "COVID":          ("2020-02-01", "2020-12-31"),
    "Tariffs":        ("2025-01-01", "2025-12-31"),
}

# Key event dates used for the vertical markers / event studies.
EVENT_DATES = {
    "GFC (Lehman)":         pd.Timestamp("2008-09-15"),
    "Euro-zone (Aug 2011)": pd.Timestamp("2011-08-01"),
    "Vol-mageddon":         pd.Timestamp("2018-02-05"),
    "COVID crash":          pd.Timestamp("2020-03-11"),
    "Liberation Day":       pd.Timestamp("2025-04-02"),
}


# ---------------------------------------------------------------------------
# 1. Slicing the panel
# ---------------------------------------------------------------------------
def subset_panel(panel, start, end):
    """Return the rows of `panel` with date in [start, end] (inclusive)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    out = panel[(panel["date"] >= start) & (panel["date"] <= end)].copy()
    return out.sort_values(["portfolio", "date"]).reset_index(drop=True)


def absorption_ratio_series(panel):
    """Universe-level absorption ratio (identical across portfolios) as a date-indexed Series."""
    one = panel[panel["portfolio"] == "min_var"][["date", "absorption_ratio"]]
    return one.set_index("date")["absorption_ratio"].sort_index()


# ---------------------------------------------------------------------------
# 2. Summary tables
# ---------------------------------------------------------------------------
def period_summary_table(panel, start, end, measures=MEASURE_COLS, stats=("mean", "std", "min", "max")):
    """
    Per-portfolio summary of each measure inside [start, end].
    Rows = portfolios (in canonical order), columns = MultiIndex (measure, stat).
    """
    sub = subset_panel(panel, start, end)
    if sub.empty:
        return pd.DataFrame()

    grouped = sub.groupby("portfolio", observed=True)[list(measures)].agg(list(stats))
    grouped = grouped.reindex([p for p in PORTFOLIO_NAMES if p in grouped.index])
    grouped.index = [PORTFOLIO_LABELS.get(p, p) for p in grouped.index]
    return grouped


def compare_windows_table(panel, windows, measure="enb_entropy", stat="mean"):
    """
    One number per (portfolio, window): the `stat` of `measure` inside each window.
    Rows = portfolios, columns = window names. Ideal for "does one method dominate in
    one regime but not another?".
    """
    cols = {}
    for name, (start, end) in windows.items():
        sub = subset_panel(panel, start, end)
        if sub.empty:
            cols[name] = pd.Series(np.nan, index=PORTFOLIO_NAMES)
            continue
        s = sub.groupby("portfolio", observed=True)[measure].agg(stat)
        cols[name] = s.reindex(PORTFOLIO_NAMES)
    table = pd.DataFrame(cols)
    table.index = [PORTFOLIO_LABELS.get(p, p) for p in table.index]
    return table


def drawdown_of_measure(panel, start, end, measure="enb_entropy"):
    """
    For each portfolio, the peak-to-trough drop of `measure` inside the window,
    reported both in level and as a percentage of the pre-trough peak.
    A large drop = the portfolio's diversification collapsed during the episode.
    """
    sub = subset_panel(panel, start, end)
    rows = []
    for name in PORTFOLIO_NAMES:
        s = sub[sub["portfolio"] == name].set_index("date")[measure].sort_index()
        if s.empty:
            continue
        running_peak = s.cummax()
        dd = s / running_peak - 1.0
        trough_date = dd.idxmin()
        rows.append({
            "portfolio": PORTFOLIO_LABELS.get(name, name),
            "window_start_level": s.iloc[0],
            "peak_level": running_peak.loc[trough_date],
            "trough_level": s.loc[trough_date],
            "trough_date": trough_date,
            "max_drawdown_pct": dd.min(),
            "window_end_level": s.iloc[-1],
        })
    return pd.DataFrame(rows).set_index("portfolio")


# ---------------------------------------------------------------------------
# 3. Event study (align measures to a shock date)
# ---------------------------------------------------------------------------
def event_study(panel, event_date, measure="enb_entropy", pre=63, post=126, normalize=True):
    """
    Align `measure` for every portfolio to trading days relative to `event_date` (t = 0).
    Returns a DataFrame indexed by integer offset (-pre ... +post), columns = portfolios.
    If normalize=True each column is rescaled to 100 at t = 0 (so the y-axis is % of the
    pre-shock diversification level).
    """
    event_date = pd.Timestamp(event_date)
    all_dates = np.array(sorted(panel["date"].unique()))
    if len(all_dates) == 0:
        return pd.DataFrame()

    # Nearest available trading day on/after the event (fallback to the closest earlier one).
    on_or_after = all_dates[all_dates >= event_date]
    anchor = on_or_after[0] if len(on_or_after) else all_dates[-1]
    anchor_i = int(np.where(all_dates == anchor)[0][0])

    lo = max(0, anchor_i - pre)
    hi = min(len(all_dates) - 1, anchor_i + post)
    window_dates = all_dates[lo:hi + 1]
    offsets = np.arange(lo, hi + 1) - anchor_i

    out = {}
    for name in PORTFOLIO_NAMES:
        s = (panel[panel["portfolio"] == name]
             .set_index("date")[measure]
             .reindex(window_dates))
        vals = s.to_numpy(dtype="float64")
        if normalize:
            base = vals[offsets == 0]
            base = base[0] if len(base) and np.isfinite(base[0]) and base[0] != 0 else np.nan
            vals = 100.0 * vals / base
        out[name] = vals

    return pd.DataFrame(out, index=pd.Index(offsets, name="days_from_event"))


# ---------------------------------------------------------------------------
# 4. Concentration of the optimized portfolios during the window
# ---------------------------------------------------------------------------
def holdings_concentration_series(portfolios, start, end, name="min_var", thresh=1e-6):
    """
    Number of non-trivial holdings (weight > thresh) of a given optimized portfolio
    through the window. A sharp drop = the optimizer piled into a handful of names.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    rows = []
    for t, port_t in portfolios.items():
        t = pd.Timestamp(t)
        if not (start <= t <= end):
            continue
        w = port_t.get(name)
        if w is None:
            continue
        arr = np.array(list(w.values()), dtype="float64")
        rows.append((t, int((arr > thresh).sum())))
    if not rows:
        return pd.Series(dtype="float64")
    idx, vals = zip(*sorted(rows))
    return pd.Series(vals, index=pd.DatetimeIndex(idx), name=f"{name}_n_holdings")


# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------
def plot_period_measures(panel, start, end, event_dates=None, portfolios_to_plot=None,
                         shade=None, title=None):
    """
    3-panel time series (one per measure) over the window, one line per portfolio.
    event_dates : dict {label: Timestamp} drawn as dashed verticals.
    shade       : optional (shade_start, shade_end) grey band (e.g. the acute selloff).
    """
    portfolios_to_plot = portfolios_to_plot or PORTFOLIO_NAMES
    sub = subset_panel(panel, start, end)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    for ax, col in zip(axes, MEASURE_COLS):
        for name in portfolios_to_plot:
            s = sub[sub["portfolio"] == name]
            ax.plot(s["date"], s[col], label=PORTFOLIO_LABELS.get(name, name), linewidth=1.1)
        ax.set_ylabel(MEASURE_LABELS[col])
        ax.grid(alpha=0.3)
        if col == "top_10_var_explained":
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        if shade is not None:
            ax.axvspan(pd.Timestamp(shade[0]), pd.Timestamp(shade[1]), color="grey", alpha=0.15)
        if event_dates:
            for lbl, d in event_dates.items():
                d = pd.Timestamp(d)
                if pd.Timestamp(start) <= d <= pd.Timestamp(end):
                    ax.axvline(d, color="firebrick", linestyle="--", alpha=0.6)
                    ax.text(d, 0.98, lbl, transform=ax.get_xaxis_transform(),
                            rotation=90, va="top", ha="right", fontsize=8)
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("Date")
    fig.suptitle(title or f"Diversification measures: {start} to {end}")
    fig.tight_layout()
    return fig, axes


def plot_event_study(panel, event_date, measure="enb_entropy", pre=63, post=126,
                     normalize=True, portfolios_to_plot=None, title=None):
    """Line plot of the event_study() output (x = trading days from the shock)."""
    portfolios_to_plot = portfolios_to_plot or PORTFOLIO_NAMES
    es = event_study(panel, event_date, measure=measure, pre=pre, post=post, normalize=normalize)

    fig, ax = plt.subplots(figsize=(11, 5))
    for name in portfolios_to_plot:
        if name in es.columns:
            ax.plot(es.index, es[name], label=PORTFOLIO_LABELS.get(name, name), linewidth=1.3)
    ax.axvline(0, color="firebrick", linestyle="--", alpha=0.7)
    if normalize:
        ax.axhline(100, color="black", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(f"{MEASURE_LABELS.get(measure, measure)} (t=0 → 100)")
    else:
        ax.set_ylabel(MEASURE_LABELS.get(measure, measure))
    ax.set_xlabel("Trading days from event")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(title or f"Event study around {pd.Timestamp(event_date).date()}: {MEASURE_LABELS.get(measure, measure)}")
    fig.tight_layout()
    return fig, ax


def plot_absorption_with_measure(panel, start, end, measure="enb_entropy",
                                 portfolios_to_plot=("min_var", "max_div"),
                                 event_dates=None, title=None):
    """
    Twin-axis plot: diversification of the high-ENB portfolios (left) against the
    universe-level absorption ratio (right). Shows the mechanical link between a
    market-wide factor spike and the portfolios' loss of diversification.
    """
    sub = subset_panel(panel, start, end)
    ar = absorption_ratio_series(panel).loc[pd.Timestamp(start):pd.Timestamp(end)]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    for name in portfolios_to_plot:
        s = sub[sub["portfolio"] == name]
        ax1.plot(s["date"], s[measure], label=f"{PORTFOLIO_LABELS.get(name, name)} — {MEASURE_LABELS.get(measure, measure)}",
                 linewidth=1.2)
    ax1.set_ylabel(MEASURE_LABELS.get(measure, measure))
    ax1.legend(loc="upper left", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(ar.index, ar.values, color="black", alpha=0.45, label="Absorption ratio (top 10)")
    ax2.set_ylabel("Absorption ratio")
    ax2.legend(loc="upper right", fontsize=8)

    if event_dates:
        for lbl, d in event_dates.items():
            d = pd.Timestamp(d)
            if pd.Timestamp(start) <= d <= pd.Timestamp(end):
                ax1.axvline(d, color="firebrick", linestyle="--", alpha=0.6)
                ax1.text(d, 0.98, lbl, transform=ax1.get_xaxis_transform(),
                         rotation=90, va="top", ha="right", fontsize=8)
    ax1.set_title(title or f"High-ENB portfolios vs. absorption ratio: {start} to {end}")
    fig.tight_layout()
    return fig, (ax1, ax2)
