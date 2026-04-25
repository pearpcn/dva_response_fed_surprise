#!/usr/bin/env python3
"""
Prepare Thailand REER controls and merge into the main Fed-shock + DVA panel.

Pipeline:
1) Source REER data from BIS webpage (auto-download) or local eer.csv.
2) Filter to Thailand, Real, Broad index.
3) Convert monthly REER to annual average.
4) Build macro controls with lag-1 terms for REER, PPI, GDP growth.
5) Merge into main dataset (Fed shocks + sectoral DVA) for 1998-2022.

Outputs:
- data_raw/reer_th_broad_real_monthly.csv
- data_raw/reer_th_broad_real_annual.csv
- data_raw/macro_controls_with_lags_annual.csv
- fed_dva_analysis/outputs/panel_main_with_reer_controls.csv
"""

from __future__ import annotations

import argparse
import io
import re
from pathlib import Path
from typing import Iterable, Optional
import zipfile

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
DATA_RAW = BASE_DIR / "data_raw"
BIS_URL = "https://www.bis.org/statistics/eer.htm"
BIS_BULK_EER_ZIP_URL = "https://data.bis.org/static/bulk/WS_EER_csv_flat.zip"

DVA_PANEL_DEFAULT = DATA_RAW / "sectoral_dva_panel.csv"
FED_SHOCK_DEFAULT = BASE_DIR / "fed_shocks_main_1998_2022.csv"
CONTROLS_DEFAULT = DATA_RAW / "controls_annual.csv"

REER_MONTHLY_OUT = DATA_RAW / "reer_th_broad_real_monthly.csv"
REER_ANNUAL_OUT = DATA_RAW / "reer_th_broad_real_annual.csv"
MACRO_OUT = DATA_RAW / "macro_controls_with_lags_annual.csv"
PANEL_OUT = BASE_DIR / "fed_dva_analysis" / "outputs" / "panel_main_with_reer_controls.csv"


def _pick_column(cols: Iterable[str], candidates: list[str]) -> Optional[str]:
    lower_to_orig = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_to_orig:
            return lower_to_orig[cand.lower()]
    return None


def _find_time_column(df: pd.DataFrame) -> str:
    candidates = ["TIME_PERIOD", "time_period", "date", "Date", "TIME", "time", "month", "Month"]
    col = _pick_column(df.columns, candidates)
    if col:
        return col

    # Fallback: first column that can be parsed as date for most rows.
    for c in df.columns:
        parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notna().mean() > 0.8:
            return c

    raise ValueError("Cannot identify time column in EER dataset.")


def _find_value_column(df: pd.DataFrame) -> str:
    candidates = ["OBS_VALUE", "obs_value", "value", "Value", "obs", "OBS"]
    col = _pick_column(df.columns, candidates)
    if col:
        return col

    # Fallback: numeric-like column with most non-null values.
    best_col = None
    best_score = -1.0
    for c in df.columns:
        numeric = pd.to_numeric(df[c], errors="coerce")
        score = numeric.notna().mean()
        if score > best_score:
            best_score = score
            best_col = c

    if best_col is None or best_score < 0.3:
        raise ValueError("Cannot identify value column in EER dataset.")
    return best_col


def _download_bis_eer_csv(out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    page = requests.get(BIS_URL, timeout=30)
    page.raise_for_status()

    # Find likely CSV links on BIS EER page.
    hrefs = re.findall(r'href="([^"]+)"', page.text, flags=re.IGNORECASE)
    csv_links = []
    for h in hrefs:
        h_low = h.lower()
        if "eer" in h_low and ".csv" in h_low:
            if h.startswith("http"):
                csv_links.append(h)
            else:
                csv_links.append("https://www.bis.org" + h)

    if csv_links:
        csv_url = csv_links[0]
        resp = requests.get(csv_url, timeout=60)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        return out_path

    # Fallback: BIS data portal bulk file (works even when EER page is JS-rendered).
    resp = requests.get(BIS_BULK_EER_ZIP_URL, timeout=120)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(
                "Auto-download failed: bulk EER zip contained no CSV file. "
                "Please download eer.csv manually and pass --eer-csv <path>."
            )
        with zf.open(csv_names[0]) as f:
            out_path.write_bytes(f.read())

    return out_path


def _read_eer_csv(path: Path) -> pd.DataFrame:
    # Prefer required columns to keep memory usage lower on BIS bulk files.
    preferred = {
        "REF_AREA:Reference area",
        "EER_TYPE:Type",
        "EER_BASKET:Basket",
        "TIME_PERIOD:Time period or range",
        "OBS_VALUE:Observation Value",
        "REF_AREA",
        "EER_TYPE",
        "EER_BASKET",
        "TIME_PERIOD",
        "OBS_VALUE",
        "date",
        "value",
    }

    try:
        return pd.read_csv(path, usecols=lambda c: c in preferred, low_memory=False)
    except Exception:
        return pd.read_csv(path, low_memory=False)


def load_eer_source(eer_csv: Optional[Path], download_if_missing: bool = True) -> pd.DataFrame:
    if eer_csv is None:
        eer_csv = DATA_RAW / "eer.csv"

    if eer_csv.exists():
        return _read_eer_csv(eer_csv)

    if download_if_missing:
        downloaded = _download_bis_eer_csv(eer_csv)
        return _read_eer_csv(downloaded)

    raise FileNotFoundError(f"EER source not found: {eer_csv}")


def filter_th_real_broad(df: pd.DataFrame) -> pd.DataFrame:
    # Prefer explicit BIS columns when available.
    area_col = _pick_column(df.columns, ["REF_AREA:Reference area", "REF_AREA", "reference area"])
    type_col = _pick_column(df.columns, ["EER_TYPE:Type", "EER_TYPE", "type"])
    basket_col = _pick_column(df.columns, ["EER_BASKET:Basket", "EER_BASKET", "basket"])

    if area_col and type_col and basket_col:
        area = df[area_col].fillna("").astype(str).str.lower()
        etype = df[type_col].fillna("").astype(str).str.lower()
        basket = df[basket_col].fillna("").astype(str).str.lower()

        mask_th = area.str.contains(r"\bth\b|thailand|tha", regex=True)
        mask_real = etype.str.contains(r"\breal\b|\br:\s*real\b", regex=True)
        mask_broad = basket.str.contains(r"\bbroad\b|\bb:\s*broad\b", regex=True)
    else:
        # Generic fallback for non-BIS-formatted files.
        object_cols = [c for c in df.columns if df[c].dtype == "object"]
        if not object_cols:
            raise ValueError("No text columns found in EER dataset for filtering.")

        text_all = df[object_cols].fillna("").astype(str).agg(" | ".join, axis=1).str.lower()
        mask_th = text_all.str.contains(r"\bth\b|thailand|tha", regex=True)
        mask_real = text_all.str.contains(r"\breal\b", regex=True)
        mask_broad = text_all.str.contains(r"\bbroad\b", regex=True)

    out = df[mask_th & mask_real & mask_broad].copy()
    if out.empty:
        raise ValueError(
            "Filtering returned 0 rows. Check whether eer.csv contains Thailand + Real + Broad labels."
        )

    return out


def annualize_reer(df_filtered: pd.DataFrame) -> pd.DataFrame:
    time_col = _find_time_column(df_filtered)
    value_col = _find_value_column(df_filtered)

    out = df_filtered.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=[time_col, value_col]).copy()

    out["year"] = out[time_col].dt.year
    # In case multiple series remain after filtering, average within each month first.
    out["month"] = out[time_col].dt.to_period("M").dt.to_timestamp()
    monthly = out.groupby("month", as_index=False)[value_col].mean()
    monthly["year"] = monthly["month"].dt.year
    monthly = monthly.rename(columns={value_col: "REER_monthly"})

    annual = monthly.groupby("year", as_index=False)["REER_monthly"].mean()
    annual = annual.rename(columns={"REER_monthly": "REER_annual"})
    return monthly, annual


def fetch_worldbank_gdp_growth_th() -> pd.DataFrame:
    """Fetch Thailand GDP growth (annual %) from World Bank API."""
    url = (
        "https://api.worldbank.org/v2/country/THA/indicator/NY.GDP.MKTP.KD.ZG"
        "?format=json&per_page=200"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    payload = resp.json()
    if not isinstance(payload, list) or len(payload) < 2:
        raise RuntimeError("Unexpected World Bank API response format.")

    rows = payload[1]
    records = []
    for r in rows:
        y = r.get("date")
        v = r.get("value")
        if y is None:
            continue
        try:
            y = int(y)
        except Exception:
            continue
        records.append({"year": y, "GDP_growth": v})

    gdp = pd.DataFrame(records)
    gdp["GDP_growth"] = pd.to_numeric(gdp["GDP_growth"], errors="coerce")
    gdp = gdp.dropna(subset=["GDP_growth"]).sort_values("year").reset_index(drop=True)
    return gdp


def build_macro_with_lags(
    controls_path: Path,
    reer_annual: pd.DataFrame,
    gdp_growth_override: Optional[Path] = None,
) -> pd.DataFrame:
    controls = pd.read_csv(controls_path)
    controls["year"] = pd.to_numeric(controls["year"], errors="coerce").astype("Int64")
    controls = controls.dropna(subset=["year"]).copy()
    controls["year"] = controls["year"].astype(int)

    # PPI: support multiple naming conventions.
    ppi_col = _pick_column(controls.columns, ["PPI_annual", "PPI", "ppi_annual", "ppi"])
    if ppi_col is None:
        raise ValueError("Cannot find PPI column in controls_annual.csv")

    base = controls[["year", ppi_col]].rename(columns={ppi_col: "PPI_annual"}).copy()

    if gdp_growth_override is not None and gdp_growth_override.exists():
        gdp = pd.read_csv(gdp_growth_override)
        year_col = _pick_column(gdp.columns, ["year", "Year"])
        gdp_col = _pick_column(gdp.columns, ["GDP_growth", "gdp_growth", "gdp_growth_pct", "NY_GDP_MKTP_KD_ZG"])
        if year_col is None or gdp_col is None:
            raise ValueError("GDP file must have year and GDP growth columns.")
        gdp = gdp[[year_col, gdp_col]].rename(columns={year_col: "year", gdp_col: "GDP_growth"})
    else:
        gdp = fetch_worldbank_gdp_growth_th()

    macro = base.merge(reer_annual, on="year", how="left")
    macro = macro.merge(gdp, on="year", how="left")
    macro = macro.sort_values("year").reset_index(drop=True)

    for c in ["REER_annual", "PPI_annual", "GDP_growth"]:
        macro[c] = pd.to_numeric(macro[c], errors="coerce")
        macro[f"{c}_lag1"] = macro[c].shift(1)

    return macro


def validate_main_period(df: pd.DataFrame, year_col: str = "year") -> None:
    years = sorted(df[year_col].dropna().astype(int).unique().tolist())
    expected = list(range(1998, 2023))
    missing = sorted(set(expected) - set(years))
    if missing:
        raise ValueError(f"Main period 1998-2022 is incomplete. Missing years: {missing}")


def build_main_panel(
    dva_panel_path: Path,
    fed_shock_path: Path,
    macro_lag_df: pd.DataFrame,
) -> pd.DataFrame:
    dva = pd.read_csv(dva_panel_path)
    fed = pd.read_csv(fed_shock_path)

    dva["year"] = pd.to_numeric(dva["year"], errors="coerce")
    fed["year"] = pd.to_numeric(fed["year"], errors="coerce")

    dva = dva[dva["year"].between(1998, 2022)].copy()
    fed = fed[fed["year"].between(1998, 2022)].copy()
    macro = macro_lag_df[macro_lag_df["year"].between(1998, 2022)].copy()

    validate_main_period(fed)
    validate_main_period(macro)

    panel = dva.merge(fed, on="year", how="inner")
    panel = panel.merge(macro, on="year", how="left")

    validate_main_period(panel)
    return panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare REER controls and merge into main panel.")
    parser.add_argument("--eer-csv", type=Path, default=DATA_RAW / "eer.csv", help="Path to BIS eer.csv")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Disable auto-download from BIS if eer.csv is missing.",
    )
    parser.add_argument(
        "--gdp-file",
        type=Path,
        default=None,
        help="Optional local GDP growth file (year + GDP_growth). If omitted, World Bank API is used.",
    )
    parser.add_argument("--dva-panel", type=Path, default=DVA_PANEL_DEFAULT)
    parser.add_argument("--fed-shock", type=Path, default=FED_SHOCK_DEFAULT)
    parser.add_argument("--controls", type=Path, default=CONTROLS_DEFAULT)
    parser.add_argument("--out-panel", type=Path, default=PANEL_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    eer_raw = load_eer_source(args.eer_csv, download_if_missing=not args.no_download)
    eer_filtered = filter_th_real_broad(eer_raw)
    reer_monthly, reer_annual = annualize_reer(eer_filtered)

    macro = build_macro_with_lags(
        controls_path=args.controls,
        reer_annual=reer_annual,
        gdp_growth_override=args.gdp_file,
    )

    panel = build_main_panel(
        dva_panel_path=args.dva_panel,
        fed_shock_path=args.fed_shock,
        macro_lag_df=macro,
    )

    REER_MONTHLY_OUT.parent.mkdir(parents=True, exist_ok=True)
    args.out_panel.parent.mkdir(parents=True, exist_ok=True)

    reer_monthly.to_csv(REER_MONTHLY_OUT, index=False)
    reer_annual.to_csv(REER_ANNUAL_OUT, index=False)
    macro.to_csv(MACRO_OUT, index=False)
    panel.to_csv(args.out_panel, index=False)

    print("Saved outputs:")
    print(f"- {REER_MONTHLY_OUT}")
    print(f"- {REER_ANNUAL_OUT}")
    print(f"- {MACRO_OUT}")
    print(f"- {args.out_panel}")
    print(f"Panel shape: {panel.shape}")
    print(f"Panel years: {int(panel['year'].min())}-{int(panel['year'].max())}")


if __name__ == "__main__":
    main()
