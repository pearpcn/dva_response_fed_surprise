#!/usr/bin/env python3
"""Step 3: Main sectoral LP interaction analysis.

This script estimates industry-level Local Projections (LP) in two designs:

1) Individual interaction: one interaction term at a time.
2) Horse-race interaction: all four interaction terms in one equation.

Baseline LP equation used by both designs:

    Delta^h log(DVA_{i,t+h}) = alpha_i
                               + beta_h * shock_t
                               + delta_h * (shock_t x Z_i)
                               + gamma' X_{t-1}
                               + psi' D_t
                               + e_{i,t+h}

where Z_i is one of (for individual design):
- BL_norm
- FL_norm
- ExportIntensity
- ImportIntensity

Main sample: 1998-2022
Appendix sample: 1995-1997 (robustness code path kept separate)

Inference:
- Sector fixed effects via within transformation
- Cluster-robust SE by sector_id

Outputs:
- fed_dva_analysis/outputs/results/sector_lp_interactions_main_*.csv
- fed_dva_analysis/outputs/results/sector_lp_interactions_appendix_*.csv
- fed_dva_analysis/outputs/results/sector_lp_interactions_main_comparison.csv
- fed_dva_analysis/outputs/results/sector_lp_horserace_main.csv
- fed_dva_analysis/outputs/results/sector_lp_horserace_appendix.csv
- fed_dva_analysis/outputs/results/sector_aggregation_scheme.csv
- fed_dva_analysis/outputs/figures/irf_interactions_high_low_main.png
"""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.sandwich_covariance import cov_cluster


warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
DATA_RAW = BASE / "data_raw"
OUT = BASE / "fed_dva_analysis" / "outputs"
RESULTS = OUT / "results"
FIGS = OUT / "figures"

MAIN_PANEL = OUT / "panel_main_with_reer_controls.csv"
APPENDIX_SHOCK = BASE / "fed_shocks_appendix_1995_1997.csv"
MACRO = DATA_RAW / "macro_controls_with_lags_annual.csv"
DVA = DATA_RAW / "sectoral_dva_panel.csv"
SECTOR_MASTER = OUT / "sector_master_table.csv"

HORIZONS = list(range(7))
SHOCK = "pure_ME_shock_annual"
CONTROLS = [
    "REER_annual_lag1",
    "PPI_annual_lag1",
    "GDP_growth_lag1",
    "dummy_dotcom",
    "dummy_gfc",
    "dummy_taper",
    "dummy_covid",
]

Z_VARS = {
    "BL": "BL_norm",
    "FL": "FL_norm",
    "Export": "ExportIntensity",
    "Import": "ImportIntensity",
}

HORSE_RACE_TERMS = {
    "int_bl": "BL_norm",
    "int_fl": "FL_norm",
    "int_export": "ExportIntensity",
    "int_import": "ImportIntensity",
}


def load_structural_characteristics() -> pd.DataFrame:
    master = pd.read_csv(SECTOR_MASTER)
    struct = master[["sector", "bwd_norm", "fwd_norm", "export_content_share", "import_content_share"]].copy()
    struct = struct.rename(
        columns={
            "sector": "sector_id",
            "bwd_norm": "BL_norm",
            "fwd_norm": "FL_norm",
            "export_content_share": "ExportIntensity",
            "import_content_share": "ImportIntensity",
        }
    )
    # Sector T (households): BL_norm=FL_norm=ExportIntensity=ImportIntensity=0
    # Kept in sample; interaction terms for T will be zero by construction.
    return struct


def build_sector_aggregation(struct: pd.DataFrame) -> pd.DataFrame:
    out = struct.copy()

    bl_cut = out["BL_norm"].median()
    fl_cut = out["FL_norm"].median()
    ex_cut = out["ExportIntensity"].median()
    im_cut = out["ImportIntensity"].median()

    out["BL_group"] = np.where(out["BL_norm"] > bl_cut, "High BL", "Low BL")
    out["FL_group"] = np.where(out["FL_norm"] > fl_cut, "High FL", "Low FL")
    out["Export_group"] = np.where(out["ExportIntensity"] > ex_cut, "High Export", "Low Export")
    out["Import_group"] = np.where(out["ImportIntensity"] > im_cut, "High Import", "Low Import")

    out["network_quadrant"] = np.select(
        [
            (out["BL_norm"] >= 1) & (out["FL_norm"] >= 1),
            (out["BL_norm"] < 1) & (out["FL_norm"] >= 1),
            (out["BL_norm"] >= 1) & (out["FL_norm"] < 1),
        ],
        ["Q1 Key", "Q2 Forward", "Q4 Backward"],
        default="Q3 Weak",
    )

    out.to_csv(RESULTS / "sector_aggregation_scheme.csv", index=False)
    return out


def load_main_panel(struct: pd.DataFrame) -> pd.DataFrame:
    panel = pd.read_csv(MAIN_PANEL)
    panel = panel[panel["year"].between(1998, 2022)].copy()
    for col, years in {
        "dummy_dotcom": [2000, 2001],
        "dummy_gfc": [2008, 2009],
        "dummy_taper": [2013],
        "dummy_covid": [2020, 2021],
    }.items():
        if col not in panel.columns:
            panel[col] = panel["year"].isin(years).astype(int)
    panel = panel.merge(struct, on="sector_id", how="left")
    panel["log_dva"] = np.log(pd.to_numeric(panel["dva"], errors="coerce").clip(lower=1))
    panel = panel.dropna(subset=["log_dva"] + list(Z_VARS.values())).copy()
    return panel


def load_appendix_panel(struct: pd.DataFrame) -> pd.DataFrame:
    dva = pd.read_csv(DVA)
    shocks = pd.read_csv(APPENDIX_SHOCK)
    macro = pd.read_csv(MACRO)

    dva = dva[dva["year"].between(1995, 1997)].copy()
    shocks = shocks[shocks["year"].between(1995, 1997)].copy()
    macro = macro[macro["year"].between(1995, 1997)].copy()

    panel = dva.merge(shocks, on="year", how="inner")
    panel = panel.merge(macro, on="year", how="left")
    for col, years in {
        "dummy_dotcom": [2000, 2001],
        "dummy_gfc": [2008, 2009],
        "dummy_taper": [2013],
        "dummy_covid": [2020, 2021],
    }.items():
        if col not in panel.columns:
            panel[col] = panel["year"].isin(years).astype(int)
    panel = panel.merge(struct, on="sector_id", how="left")
    panel["log_dva"] = np.log(pd.to_numeric(panel["dva"], errors="coerce").clip(lower=1))
    panel = panel.dropna(subset=["log_dva"] + list(Z_VARS.values())).copy()
    return panel


def within_transform(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = out[col] - out.groupby("sector_id")[col].transform("mean")
    return out


def prepare_horizon_data(df: pd.DataFrame, h: int, z_col: str) -> pd.DataFrame:
    work = df.sort_values(["sector_id", "year"]).copy()
    work["log_dva_lead"] = work.groupby("sector_id")["log_dva"].shift(-h)
    work["log_dva_lag1"] = work.groupby("sector_id")["log_dva"].shift(1)
    work["dy"] = work["log_dva_lead"] - work["log_dva_lag1"]
    work["interaction"] = work[SHOCK] * work[z_col]

    needed = ["dy", SHOCK, "interaction", z_col, "sector_id", "year"] + CONTROLS
    work = work.dropna(subset=needed).copy()
    return work


def estimate_one_horizon(df_h: pd.DataFrame, h: int, z_label: str, z_col: str) -> dict:
    regressors = [SHOCK, "interaction"] + CONTROLS

    min_obs = len(regressors) + 8
    if len(df_h) < min_obs or df_h["sector_id"].nunique() < 5:
        return {
            "h": h,
            "z_label": z_label,
            "z_col": z_col,
            "coef_shock": np.nan,
            "se_shock": np.nan,
            "p_shock": np.nan,
            "coef_interaction": np.nan,
            "se_interaction": np.nan,
            "p_interaction": np.nan,
            "ci95_lo_interaction": np.nan,
            "ci95_hi_interaction": np.nan,
            "nobs": len(df_h),
            "n_sectors": df_h["sector_id"].nunique(),
            "status": "insufficient_observations",
        }

    demean_cols = ["dy"] + regressors
    dm = within_transform(df_h, demean_cols)
    y = dm["dy"]
    X = dm[regressors]

    fit = sm.OLS(y, X).fit()
    cov = cov_cluster(fit, df_h["sector_id"])
    se = pd.Series(np.sqrt(np.diag(cov)), index=regressors)

    dof = max(len(df_h) - len(regressors), 1)
    t_shock = fit.params[SHOCK] / se[SHOCK] if se[SHOCK] > 0 else np.nan
    t_int = fit.params["interaction"] / se["interaction"] if se["interaction"] > 0 else np.nan
    p_shock = 2 * (1 - stats.t.cdf(abs(t_shock), df=dof)) if not np.isnan(t_shock) else np.nan
    p_int = 2 * (1 - stats.t.cdf(abs(t_int), df=dof)) if not np.isnan(t_int) else np.nan
    crit95 = stats.t.ppf(0.975, dof)

    cov_df = pd.DataFrame(cov, index=regressors, columns=regressors)

    return {
        "h": h,
        "z_label": z_label,
        "z_col": z_col,
        "coef_shock": float(fit.params[SHOCK]),
        "se_shock": float(se[SHOCK]),
        "p_shock": float(p_shock),
        "coef_interaction": float(fit.params["interaction"]),
        "se_interaction": float(se["interaction"]),
        "p_interaction": float(p_int),
        "ci95_lo_interaction": float(fit.params["interaction"] - crit95 * se["interaction"]),
        "ci95_hi_interaction": float(fit.params["interaction"] + crit95 * se["interaction"]),
        "cov_shock_interaction": float(cov_df.loc[SHOCK, "interaction"]),
        "var_shock": float(cov_df.loc[SHOCK, SHOCK]),
        "var_interaction": float(cov_df.loc["interaction", "interaction"]),
        "nobs": len(df_h),
        "n_sectors": int(df_h["sector_id"].nunique()),
        "status": "ok",
    }


def run_sample(panel: pd.DataFrame, sample_name: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}

    for z_label, z_col in Z_VARS.items():
        rows = []
        for h in HORIZONS:
            df_h = prepare_horizon_data(panel, h, z_col)
            rows.append(estimate_one_horizon(df_h, h, z_label, z_col))
        out = pd.DataFrame(rows)
        outputs[z_label] = out
        out.to_csv(RESULTS / f"sector_lp_interactions_{sample_name}_{z_label.lower()}.csv", index=False)

    comparison_rows = []
    for h in HORIZONS:
        row = {"h": h}
        for z_label, out in outputs.items():
            sub = out.loc[out["h"] == h].iloc[0]
            row[f"delta_{z_label.lower()}"] = sub["coef_interaction"]
            row[f"p_{z_label.lower()}"] = sub["p_interaction"]
            row[f"se_{z_label.lower()}"] = sub["se_interaction"]
        comparison_rows.append(row)

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(RESULTS / f"sector_lp_interactions_{sample_name}_comparison.csv", index=False)
    return outputs, comparison


def prepare_horizon_data_horserace(df: pd.DataFrame, h: int) -> pd.DataFrame:
    work = df.sort_values(["sector_id", "year"]).copy()
    work["log_dva_lead"] = work.groupby("sector_id")["log_dva"].shift(-h)
    work["log_dva_lag1"] = work.groupby("sector_id")["log_dva"].shift(1)
    work["dy"] = work["log_dva_lead"] - work["log_dva_lag1"]

    for int_name, z_col in HORSE_RACE_TERMS.items():
        work[int_name] = work[SHOCK] * work[z_col]

    needed = ["dy", SHOCK, "sector_id", "year"] + CONTROLS + list(HORSE_RACE_TERMS.keys())
    work = work.dropna(subset=needed).copy()
    return work


def estimate_horserace_horizon(df_h: pd.DataFrame, h: int) -> dict:
    regressors = [SHOCK] + list(HORSE_RACE_TERMS.keys()) + CONTROLS

    min_obs = len(regressors) + 8
    if len(df_h) < min_obs or df_h["sector_id"].nunique() < 5:
        out = {"h": h, "nobs": len(df_h), "n_sectors": int(df_h["sector_id"].nunique()), "status": "insufficient_observations"}
        for v in [SHOCK] + list(HORSE_RACE_TERMS.keys()):
            out[f"coef_{v}"] = np.nan
            out[f"se_{v}"] = np.nan
            out[f"p_{v}"] = np.nan
        return out

    dm = within_transform(df_h, ["dy"] + regressors)
    y = dm["dy"]
    X = dm[regressors]

    fit = sm.OLS(y, X).fit()
    cov = cov_cluster(fit, df_h["sector_id"])
    se = pd.Series(np.sqrt(np.diag(cov)), index=regressors)

    dof = max(len(df_h) - len(regressors), 1)
    out = {"h": h, "nobs": len(df_h), "n_sectors": int(df_h["sector_id"].nunique()), "status": "ok"}

    for v in [SHOCK] + list(HORSE_RACE_TERMS.keys()):
        coef = float(fit.params[v])
        sev = float(se[v])
        tval = coef / sev if sev > 0 else np.nan
        pval = 2 * (1 - stats.t.cdf(abs(tval), df=dof)) if not np.isnan(tval) else np.nan
        out[f"coef_{v}"] = coef
        out[f"se_{v}"] = sev
        out[f"p_{v}"] = pval

    return out


def run_horserace_sample(panel: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        df_h = prepare_horizon_data_horserace(panel, h)
        rows.append(estimate_horserace_horizon(df_h, h))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / f"sector_lp_horserace_{sample_name}.csv", index=False)
    return out


def _group_effects(results: pd.DataFrame, z_mean: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    beta = results["coef_shock"].to_numpy()
    delta = results["coef_interaction"].to_numpy()
    effect = beta + delta * z_mean

    var = (
        results["var_shock"].to_numpy()
        + (z_mean ** 2) * results["var_interaction"].to_numpy()
        + 2 * z_mean * results["cov_shock_interaction"].to_numpy()
    )
    se = np.sqrt(np.clip(var, 0, None))
    lo = effect - 1.96 * se
    hi = effect + 1.96 * se
    return effect * 100, lo * 100, hi * 100


def plot_high_low(panel: pd.DataFrame, outputs: dict[str, pd.DataFrame], sample_name: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="white")
    axes = axes.flatten()

    for ax, (z_label, z_col) in zip(axes, Z_VARS.items()):
        sector_vals = panel[["sector_id", z_col]].drop_duplicates().dropna()
        cutoff = sector_vals[z_col].median()
        low_mean = sector_vals.loc[sector_vals[z_col] <= cutoff, z_col].mean()
        high_mean = sector_vals.loc[sector_vals[z_col] > cutoff, z_col].mean()

        res = outputs[z_label]
        ok = res[res["status"] == "ok"].copy()
        if ok.empty:
            ax.text(0.5, 0.5, "Insufficient observations", ha="center", va="center")
            ax.set_title(f"{z_label}: High vs Low")
            continue

        high, high_lo, high_hi = _group_effects(ok, high_mean)
        low, low_lo, low_hi = _group_effects(ok, low_mean)
        h = ok["h"].to_numpy()

        ax.fill_between(h, high_lo, high_hi, color="#c6dbef", alpha=0.35)
        ax.fill_between(h, low_lo, low_hi, color="#fdd0a2", alpha=0.30)
        ax.plot(h, high, color="#08519c", marker="o", linewidth=2, label=f"High {z_label}")
        ax.plot(h, low, color="#d94801", marker="s", linewidth=2, label=f"Low {z_label}")
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{z_label}: High vs Low", fontsize=11, fontweight="bold")
        ax.set_xlabel("Horizon h")
        ax.set_ylabel("Effect on DVA (%)")
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        ax.legend(frameon=False, fontsize=9)

    fig.suptitle(f"Interaction LP IRFs by High vs Low Industry Characteristic ({sample_name})", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / f"irf_interactions_high_low_{sample_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    struct = load_structural_characteristics()
    build_sector_aggregation(struct)
    main_panel = load_main_panel(struct)
    appendix_panel = load_appendix_panel(struct)

    main_outputs, _ = run_sample(main_panel, "main")
    plot_high_low(main_panel, main_outputs, "main")
    run_horserace_sample(main_panel, "main")

    appendix_outputs, _ = run_sample(appendix_panel, "appendix")
    plot_high_low(appendix_panel, appendix_outputs, "appendix")
    run_horserace_sample(appendix_panel, "appendix")

    print("Saved outputs:")
    for sample in ["main", "appendix"]:
        for z_label in Z_VARS:
            print(f"- {RESULTS / f'sector_lp_interactions_{sample}_{z_label.lower()}.csv'}")
        print(f"- {RESULTS / f'sector_lp_interactions_{sample}_comparison.csv'}")
        print(f"- {RESULTS / f'sector_lp_horserace_{sample}.csv'}")
        print(f"- {FIGS / f'irf_interactions_high_low_{sample}.png'}")
    print(f"- {RESULTS / 'sector_aggregation_scheme.csv'}")


if __name__ == "__main__":
    main()