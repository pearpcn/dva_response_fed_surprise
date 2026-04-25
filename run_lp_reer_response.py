#!/usr/bin/env python3
"""Estimate Local Projections for Thailand REER response to US Fed shocks.

Data source:
- fed_dva_analysis/outputs/panel_main_with_reer_controls.csv

Specification:
- Dependent variable: log(REER_{t+h}) - log(REER_{t-1})
- Main regressor: pure_ME_shock_annual_t
- Controls: PPI_annual_lag1, GDP_growth_lag1
- Crisis dummies at time t: dummy_dotcom, dummy_gfc, dummy_taper, dummy_covid
- Inference: Newey-West HAC with nlags = h + 1

Outputs:
- fed_dva_analysis/outputs/results/reer_lp_results.csv
- fed_dva_analysis/outputs/figures/reer_lp_irf.png
"""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.tools.tools import add_constant


warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
INPUT = BASE / "fed_dva_analysis" / "outputs" / "panel_main_with_reer_controls.csv"
OUT_DIR = BASE / "fed_dva_analysis" / "outputs"
RESULTS_DIR = OUT_DIR / "results"
FIG_DIR = OUT_DIR / "figures"
RESULTS_PATH = RESULTS_DIR / "reer_lp_results.csv"
FIG_PATH = FIG_DIR / "reer_lp_irf.png"

HORIZONS = list(range(7))
SHOCK = "pure_ME_shock_annual"
CONTROLS = [
    "PPI_annual_lag1",
    "GDP_growth_lag1",
    "dummy_dotcom",
    "dummy_gfc",
    "dummy_taper",
    "dummy_covid",
]


def load_annual_series() -> pd.DataFrame:
    df = pd.read_csv(INPUT)
    keep = ["year", "REER_annual", SHOCK] + CONTROLS
    annual = df[keep].drop_duplicates(subset=["year"]).sort_values("year").reset_index(drop=True)
    annual = annual[annual["year"].between(1998, 2022)].copy()

    expected_years = list(range(1998, 2023))
    found_years = annual["year"].astype(int).tolist()
    if found_years != expected_years:
        raise ValueError(f"Expected continuous years 1998-2022, got {found_years}")

    annual["log_reer"] = np.log(pd.to_numeric(annual["REER_annual"], errors="coerce"))
    return annual


def run_lp(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    for h in HORIZONS:
        work = df.copy()
        work[f"log_reer_lead_{h}"] = work["log_reer"].shift(-h)
        work["log_reer_lag1"] = work["log_reer"].shift(1)
        work["dy_h"] = work[f"log_reer_lead_{h}"] - work["log_reer_lag1"]

        reg_cols = ["dy_h", SHOCK] + CONTROLS
        reg = work[reg_cols].dropna().copy()
        x = add_constant(reg[[SHOCK] + CONTROLS], has_constant="add")
        y = reg["dy_h"]

        fit = OLS(y, x).fit()
        nw_lags = max(h + 1, 1)
        cov = cov_hac(fit, nlags=nw_lags, use_correction=True)
        se = float(np.sqrt(np.diag(cov))[list(x.columns).index(SHOCK)])
        coef = float(fit.params[SHOCK])
        tstat = coef / se if se > 0 else np.nan
        dof = len(reg) - len(x.columns)
        pval = 2 * (1 - stats.t.cdf(abs(tstat), df=dof)) if dof > 0 else np.nan
        crit95 = stats.t.ppf(0.975, dof) if dof > 0 else np.nan

        rows.append(
            {
                "h": h,
                "coef_log": coef,
                "coef_pct": coef * 100,
                "se_log": se,
                "se_pct": se * 100,
                "tstat": tstat,
                "pval": pval,
                "ci95_lo_log": coef - crit95 * se,
                "ci95_hi_log": coef + crit95 * se,
                "ci95_lo_pct": (coef - crit95 * se) * 100,
                "ci95_hi_pct": (coef + crit95 * se) * 100,
                "nobs": len(reg),
                "nw_lags": nw_lags,
            }
        )

    return pd.DataFrame(rows)


def plot_irf(results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8), facecolor="white")
    ax.set_facecolor("white")

    ax.fill_between(
        results["h"],
        results["ci95_lo_pct"],
        results["ci95_hi_pct"],
        color="#bdbdbd",
        alpha=0.35,
        label="95% CI",
    )
    ax.plot(
        results["h"],
        results["coef_pct"],
        color="#1f4e79",
        linewidth=2.2,
        marker="o",
        markersize=6,
        label="IRF of REER",
    )
    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.8)

    h0 = results.loc[results["h"] == 0].iloc[0]
    annotation = f"h=0: coef = {h0['coef_pct']:.2f}% | p-value = {h0['pval']:.3f}"
    ax.text(
        0.02,
        0.96,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#666666", "boxstyle": "round,pad=0.35"},
    )

    ax.set_title("Impulse Response of Thailand REER to US Fed Shock", fontsize=13, fontweight="bold")
    ax.set_xlabel("Horizon h (years)", fontsize=11)
    ax.set_ylabel("Percentage change from t-1 (%)", fontsize=11)
    ax.set_xticks(HORIZONS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    annual = load_annual_series()
    results = run_lp(annual)
    results.to_csv(RESULTS_PATH, index=False)
    plot_irf(results)

    h0 = results.loc[results["h"] == 0].iloc[0]
    print("Saved outputs:")
    print(f"- {RESULTS_PATH}")
    print(f"- {FIG_PATH}")
    print(
        "Immediate effect (h=0): "
        f"coef = {h0['coef_log']:.6f} log points ({h0['coef_pct']:.2f}%), "
        f"p-value = {h0['pval']:.4f}"
    )


if __name__ == "__main__":
    main()