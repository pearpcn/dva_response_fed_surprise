"""
Process Fed monetary policy shocks for thesis robustness checks.

This script addresses committee feedback on three fronts:
1) Information-shock filtering:
   Uses stock-market sign restrictions (S&P 500 return response) to separate
   pure monetary policy shocks from information shocks.
2) Annual aggregation without excessive cancellation:
	Uses month-based time weights so early-year shocks receive higher annual
	influence than late-year shocks within the same year.
3) ME-focused decomposition with component transparency:
   Uses total surprise ME as the main decomposition target while still
   preserving STMT and PC for component-level follow-up.

Latest design update:
- Produces two annual outputs:
  * Main study: 1998-2022 (managed-float era)
  * Appendix:   1995-1997 (fixed-rate era)
- Adds crisis dummies (main sample only) as LP controls to absorb major
  global/systemic episodes and improve precision of the Fed-shock coefficient.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


START_DATE = "1995-01-01"
END_DATE = "2022-12-31"

# Input/output defaults
DEFAULT_INPUT_CANDIDATES = [
	Path("mps_SFFED.csv"),
	Path("data_raw/mps_SFFED.csv"),
]
ANNUAL_OUTPUT = Path("fed_shocks_cleaned_annual.csv")
MAIN_OUTPUT = Path("fed_shocks_main_1998_2022.csv")
APPENDIX_OUTPUT = Path("fed_shocks_appendix_1995_1997.csv")
SCATTER_OUTPUT = Path("me_vs_sp500_decomposition.png")
SCATTER_OUTPUT_EASY = Path("me_vs_sp500_decomposition_easy.png")
METADATA_OUTPUT = Path("fed_shocks_outputs_readme.md")

# Optional: set to True if you want STMT/PC decomposition columns too
DECOMPOSE_COMPONENTS = True


def find_input_file() -> Path:
	"""Find the shock input file from common locations."""
	for candidate in DEFAULT_INPUT_CANDIDATES:
		if candidate.exists():
			return candidate
	raise FileNotFoundError(
		"Could not find mps_SFFED.csv in working directory or data_raw/."
	)


def download_sp500_returns(start: str, end: str) -> pd.DataFrame:
	"""
	Download S&P 500 data and compute daily log returns.

	Using market returns helps classify whether a rate surprise likely reflects
	pure policy tightening/easing versus central bank information revelation.
	"""
	raw = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=False)
	if raw.empty:
		raise RuntimeError("Failed to download S&P 500 data from yfinance.")

	# yfinance can return MultiIndex columns in some versions.
	if isinstance(raw.columns, pd.MultiIndex):
		raw.columns = raw.columns.get_level_values(0)

	px_col = "Adj Close" if "Adj Close" in raw.columns else "Close"
	prices = raw[px_col].rename("sp500_price")
	rets = np.log(prices / prices.shift(1)).rename("sp500_return")

	out = rets.dropna().to_frame().reset_index()
	out = out.rename(columns={"Date": "trade_date"})
	out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.tz_localize(None)
	out = out.sort_values("trade_date").reset_index(drop=True)
	return out


def load_shocks(path: Path) -> pd.DataFrame:
	"""Load mps_SFFED data and enforce required columns."""
	shocks = pd.read_csv(path)
	required = {"Date", "STMT", "PC", "ME"}
	missing = required - set(shocks.columns)
	if missing:
		raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

	shocks = shocks.copy()
	shocks["Date"] = pd.to_datetime(shocks["Date"])
	shocks = shocks.sort_values("Date").reset_index(drop=True)
	return shocks


def merge_with_next_trading_day(
	shocks: pd.DataFrame, sp500_rets: pd.DataFrame
) -> pd.DataFrame:
	"""
	Merge shocks with stock returns.

	If an FOMC date is non-trading, merge_asof(direction='forward') picks the
	next available trading day return, matching the requested rule.
	"""
	left = shocks.rename(columns={"Date": "event_date"}).sort_values("event_date")
	right = sp500_rets.sort_values("trade_date")

	# Ensure identical datetime dtype for merge_asof keys.
	left["event_date"] = pd.to_datetime(left["event_date"]).astype("datetime64[ns]")
	right["trade_date"] = pd.to_datetime(right["trade_date"]).astype("datetime64[ns]")

	merged = pd.merge_asof(
		left,
		right,
		left_on="event_date",
		right_on="trade_date",
		direction="forward",
	)

	if merged["sp500_return"].isna().any():
		n_miss = int(merged["sp500_return"].isna().sum())
		raise ValueError(f"Could not map {n_miss} event(s) to a trading-day return.")

	merged = merged.rename(columns={"event_date": "Date"})
	merged["Date"] = pd.to_datetime(merged["Date"])
	return merged


def sign_restriction_decompose(shock: pd.Series, market_ret: pd.Series) -> tuple[pd.Series, pd.Series]:
	"""
	Decompose a surprise into pure-MP vs information-shock components.

	Pure MP sign logic:
	- Hawkish: shock > 0 and stock return < 0
	- Dovish:  shock < 0 and stock return > 0

	Same-sign responses are treated as information-shock contamination.
	"""
	pure_mask = ((shock > 0) & (market_ret < 0)) | ((shock < 0) & (market_ret > 0))
	pure = shock.where(pure_mask, 0.0)
	info = shock.where(~pure_mask, 0.0)
	return pure, info


def add_decomposition_columns(df: pd.DataFrame, decompose_components: bool = True) -> pd.DataFrame:
	"""Create decomposition columns for ME and optionally STMT/PC."""
	out = df.copy()

	out["pure_ME_shock"], out["info_shock"] = sign_restriction_decompose(
		out["ME"], out["sp500_return"]
	)

	out["classification"] = np.where(out["pure_ME_shock"] != 0, "Pure MP", "Information")
	out["pure_type"] = np.select(
		[
			(out["pure_ME_shock"] > 0) & (out["sp500_return"] < 0),
			(out["pure_ME_shock"] < 0) & (out["sp500_return"] > 0),
		],
		["Hawkish MP", "Dovish MP"],
		default="Information",
	)

	if decompose_components:
		out["pure_STMT_shock"], out["info_STMT_shock"] = sign_restriction_decompose(
			out["STMT"], out["sp500_return"]
		)
		out["pure_PC_shock"], out["info_PC_shock"] = sign_restriction_decompose(
			out["PC"], out["sp500_return"]
		)

	return out


def aggregate_annual(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Aggregate to annual frequency using committee-requested time weights.

	Weight formula:
		w_m = (12 - m + 1) / 12
	where m is event month. January gets highest weight (=1.0) and December
	gets lowest weight (=1/12), reflecting that early-year shocks have more
	time to transmit within the same calendar year.
	"""
	x = df.copy()
	x["year"] = x["Date"].dt.year
	x["month"] = x["Date"].dt.month
	x["time_weight"] = (12.0 - x["month"] + 1.0) / 12.0

	# Time-weighted annual shocks
	x["pure_ME_weighted"] = x["pure_ME_shock"] * x["time_weight"]
	x["STMT_weighted"] = x["STMT"] * x["time_weight"]
	x["PC_weighted"] = x["PC"] * x["time_weight"]

	# Committee request: annual volatility as sum of squared ME
	x["ME_sq"] = x["ME"] ** 2

	annual = (
		x.groupby("year", as_index=False)
		.agg(
			pure_ME_shock_annual=("pure_ME_weighted", "sum"),
			STMT_annual=("STMT_weighted", "sum"),
			PC_annual=("PC_weighted", "sum"),
			ME_volatility_annual=("ME_sq", "sum"),
			n_events=("ME", "size"),
		)
		.sort_values("year")
	)

	return annual


def add_main_study_dummies(main_df: pd.DataFrame) -> pd.DataFrame:
	"""
	Add crisis/event dummies for LP controls in the 1998-2022 main sample.

	These controls absorb common global disturbances (dot-com, GFC, taper,
	COVID) so the identified Fed-shock coefficient is less contaminated by
	non-policy systemic shocks.
	"""
	out = main_df.copy()
	out["dummy_dotcom"] = out["year"].isin([2000, 2001]).astype(int)
	out["dummy_gfc"] = out["year"].isin([2008, 2009]).astype(int)
	out["dummy_taper"] = (out["year"] == 2013).astype(int)
	out["dummy_covid"] = out["year"].isin([2020, 2021]).astype(int)
	return out


def validate_year_splits(main_df: pd.DataFrame, appendix_df: pd.DataFrame) -> None:
	"""Sanity checks for year-split integrity."""
	if main_df.empty:
		raise ValueError("Main sample is empty; expected years 1998-2022.")
	if appendix_df.empty:
		raise ValueError("Appendix sample is empty; expected years 1995-1997.")

	main_min, main_max = int(main_df["year"].min()), int(main_df["year"].max())
	app_min, app_max = int(appendix_df["year"].min()), int(appendix_df["year"].max())

	if (main_min, main_max) != (1998, 2022):
		raise AssertionError(
			f"Main sample year range mismatch: got {main_min}-{main_max}, expected 1998-2022"
		)
	if (app_min, app_max) != (1995, 1997):
		raise AssertionError(
			f"Appendix sample year range mismatch: got {app_min}-{app_max}, expected 1995-1997"
		)


def write_output_metadata(path: Path) -> None:
	"""Write a short README describing output files and column definitions."""
	text = r"""# Fed Shock Outputs: Data Dictionary

This file documents the CSV outputs produced by `process_fed_shocks.py`.

## Files

- `fed_shocks_main_1998_2022.csv`
  - Main study sample (managed-float era).
  - Contains annual shock aggregates and LP control dummies.
- `fed_shocks_appendix_1995_1997.csv`
  - Appendix sample (fixed-rate era).
  - Contains annual shock aggregates only.
- `fed_shocks_cleaned_annual.csv`
  - Full annual series (1995-2022), before sample split.

## Core Columns (both main and appendix)

- `year`: Calendar year.
- `pure_ME_shock_annual`: Time-weighted annual aggregate of sign-restricted pure MP shock from `ME`.
- `STMT_annual`: Time-weighted annual aggregate of target shock component (`STMT`).
- `PC_annual`: Time-weighted annual aggregate of path/forward-guidance shock component (`PC`).
- `ME_volatility_annual`: Annual shock volatility proxy, computed as sum of squared `ME` within year.
- `n_events`: Number of FOMC events in that year.

## Time-Weighting Equation (for Appendix)

Monthly event weight is defined as:

$$
w_m = \frac{12 - m + 1}{12}
$$

where $m$ is the event month.

Implications:
- January ($m=1$): $w_1 = 1.0$
- December ($m=12$): $w_{12} = 1/12$

Annual aggregation uses:

$$
Shock_t^{annual} = \sum_{m \in t} \left( Shock_{m,t} \times w_m \right)
$$

## Main-only LP Control Dummies (`fed_shocks_main_1998_2022.csv`)

- `dummy_dotcom`: 1 in 2000-2001, else 0.
- `dummy_gfc`: 1 in 2008-2009, else 0.
- `dummy_taper`: 1 in 2013, else 0.
- `dummy_covid`: 1 in 2020-2021, else 0.

These dummies are intended as controls in LP regressions to absorb major
global/systemic disturbances and improve precision of the Fed shock coefficient.
"""
	path.write_text(text, encoding="utf-8")


def make_summary_table(df: pd.DataFrame) -> pd.DataFrame:
	"""Build summary table for Pure MP vs Information classifications."""
	counts = (
		df.groupby(["classification", "pure_type"], dropna=False)
		.size()
		.reset_index(name="count")
		.sort_values(["classification", "pure_type"])
	)
	return counts


def plot_scatter(df: pd.DataFrame, out_path: Path) -> None:
	"""Scatter ME vs S&P 500 return with quadrant labels and classification colors."""
	color_map = {"Pure MP": "#1f77b4", "Information": "#d62728"}
	colors = df["classification"].map(color_map)

	plt.figure(figsize=(10, 7))
	plt.scatter(df["ME"], df["sp500_return"], c=colors, alpha=0.8, edgecolor="white", linewidth=0.5)

	plt.axhline(0, color="black", linewidth=1)
	plt.axvline(0, color="black", linewidth=1)

	# Quadrant annotations for decomposition interpretation
	plt.text(0.02, 0.02, "Q1: ME>0, r>0\nInformation shock", transform=plt.gca().transAxes, fontsize=9)
	plt.text(0.72, 0.02, "Q2: ME<0, r>0\nDovish pure MP", transform=plt.gca().transAxes, fontsize=9)
	plt.text(0.02, 0.87, "Q3: ME>0, r<0\nHawkish pure MP", transform=plt.gca().transAxes, fontsize=9)
	plt.text(0.72, 0.87, "Q4: ME<0, r<0\nInformation shock", transform=plt.gca().transAxes, fontsize=9)

	# Legend
	for label, color in color_map.items():
		plt.scatter([], [], c=color, label=label)
	plt.legend(title="Classification")

	plt.title("ME Shock vs S&P 500 Return (Sign-Restriction Decomposition)")
	plt.xlabel("ME (Total Surprise)")
	plt.ylabel("S&P 500 Log Return (next trading day if needed)")
	plt.tight_layout()
	plt.savefig(out_path, dpi=300)
	plt.close()


def plot_scatter_easy(df: pd.DataFrame, out_path: Path) -> None:
	"""Create a simpler, thesis-friendly scatter with clearly separated quadrants."""
	fig, ax = plt.subplots(figsize=(14, 8))

	# Axis limits with small margins for clean annotation placement.
	xmin, xmax = df["ME"].min(), df["ME"].max()
	ymin, ymax = df["sp500_return"].min(), df["sp500_return"].max()
	xpad = (xmax - xmin) * 0.12 if xmax > xmin else 0.1
	ypad = (ymax - ymin) * 0.12 if ymax > ymin else 0.1
	xl, xr = xmin - xpad, xmax + xpad
	yb, yt = ymin - ypad, ymax + ypad

	ax.set_xlim(xl, xr)
	ax.set_ylim(yb, yt)

	# Subtle quadrant shading (blue for pure MP quadrants, red for information quadrants).
	ax.axvspan(0, xr, ymin=(0 - yb) / (yt - yb), ymax=1, color="#fee2e2", alpha=0.25, zorder=0)  # Q1 red
	ax.axvspan(xl, 0, ymin=(0 - yb) / (yt - yb), ymax=1, color="#dbeafe", alpha=0.25, zorder=0)  # Q2 blue
	ax.axvspan(0, xr, ymin=0, ymax=(0 - yb) / (yt - yb), color="#dbeafe", alpha=0.25, zorder=0)  # Q3 blue
	ax.axvspan(xl, 0, ymin=0, ymax=(0 - yb) / (yt - yb), color="#fee2e2", alpha=0.25, zorder=0)  # Q4 red

	# Plot points by class with high contrast.
	pure = df[df["classification"] == "Pure MP"]
	info = df[df["classification"] == "Information"]
	ax.scatter(info["ME"], info["sp500_return"], s=42, c="#dc2626", alpha=0.85, edgecolor="white", linewidth=0.4, label="Information Shocks")
	ax.scatter(pure["ME"], pure["sp500_return"], s=42, c="#2563eb", alpha=0.9, edgecolor="white", linewidth=0.4, label="Pure MP Shocks")

	# Origin crosshair.
	ax.axhline(0, color="black", lw=1.4)
	ax.axvline(0, color="black", lw=1.4)

	# Compact quadrant labels.
	x_left = xl + 0.03 * (xr - xl)
	x_right = xl + 0.57 * (xr - xl)
	y_top = yb + 0.83 * (yt - yb)
	y_bottom = yb + 0.08 * (yt - yb)

	ax.text(x_right, y_top, "Q1: ECON GOOD Info\nRate(+), Stock(+)\n(Information)", fontsize=10, color="#7f1d1d", weight="bold")
	ax.text(x_left, y_top, "Q2: DOVISH Pure MP\nRate(-), Stock(+)\n(Genuine Easing)", fontsize=10, color="#1e3a8a", weight="bold")
	ax.text(x_right, y_bottom, "Q3: HAWKISH Pure MP\nRate(+), Stock(-)\n(Genuine Tightening)", fontsize=10, color="#1e3a8a", weight="bold")
	ax.text(x_left, y_bottom, "Q4: ECON BAD Info\nRate(-), Stock(-)\n(Information)", fontsize=10, color="#7f1d1d", weight="bold")

	ax.set_title("Fed Shock Identification: Pure MP vs Information (Easy-Read)", fontsize=15, weight="bold")
	ax.set_xlabel("High-Frequency Fed Shock (ME)", fontsize=12)
	ax.set_ylabel("S&P 500 High-Frequency Return", fontsize=12)
	ax.grid(alpha=0.25, linestyle="--")
	leg = ax.legend(loc="upper center", ncol=2, frameon=True, fontsize=11)
	leg.get_frame().set_alpha(0.95)

	plt.tight_layout()
	plt.savefig(out_path, dpi=320)
	plt.close()


def main() -> None:
	print("Step 1: Downloading S&P 500 and computing returns...")
	sp500 = download_sp500_returns(START_DATE, END_DATE)

	print("Step 2: Loading shock data...")
	input_path = find_input_file()
	shocks = load_shocks(input_path)

	# Keep only the analysis window to match the requested sample.
	start_ts = pd.to_datetime(START_DATE)
	end_ts = pd.to_datetime(END_DATE)
	orig_n = len(shocks)
	shocks = shocks[(shocks["Date"] >= start_ts) & (shocks["Date"] <= end_ts)].copy()
	print(f"  Using events in [{START_DATE}, {END_DATE}] -> {len(shocks)} of {orig_n} rows")

	print("Step 3: Merging shocks with next-available trading-day returns...")
	merged = merge_with_next_trading_day(shocks, sp500)

	print("Step 4: Decomposing ME into pure MP vs information shocks...")
	decomposed = add_decomposition_columns(merged, decompose_components=DECOMPOSE_COMPONENTS)

	print("Step 5: Aggregating annually with time weights...")
	annual = aggregate_annual(decomposed)

	print("Step 6: Exporting annual datasets (main + appendix)...")
	annual.to_csv(ANNUAL_OUTPUT, index=False)

	# Main study sample: managed-float era
	annual_main = annual[(annual["year"] >= 1998) & (annual["year"] <= 2022)].copy()
	annual_main = add_main_study_dummies(annual_main)
	annual_main.to_csv(MAIN_OUTPUT, index=False)

	# Appendix sample: fixed-rate era
	annual_appendix = annual[(annual["year"] >= 1995) & (annual["year"] <= 1997)].copy()
	annual_appendix.to_csv(APPENDIX_OUTPUT, index=False)

	# Sanity checks for requested year splits.
	validate_year_splits(annual_main, annual_appendix)

	# Write short output documentation.
	write_output_metadata(METADATA_OUTPUT)

	summary = make_summary_table(decomposed)
	print("\nClassification summary (events):")
	print(summary.to_string(index=False))

	print("\nStep 7: Creating scatter plot...")
	plot_scatter(decomposed, SCATTER_OUTPUT)
	plot_scatter_easy(decomposed, SCATTER_OUTPUT_EASY)

	print("\nDone.")
	print(f"Input file:  {input_path}")
	print(f"Annual CSV (full):      {ANNUAL_OUTPUT}")
	print(f"Annual CSV (main):      {MAIN_OUTPUT}")
	print(f"Annual CSV (appendix):  {APPENDIX_OUTPUT}")
	print(f"Metadata README:        {METADATA_OUTPUT}")
	print(f"Scatter PNG:            {SCATTER_OUTPUT}")
	print(f"Scatter PNG (easy):     {SCATTER_OUTPUT_EASY}")


if __name__ == "__main__":
	main()
