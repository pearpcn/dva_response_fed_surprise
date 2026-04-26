# DVA Response to U.S. Fed Monetary Surprises — Thailand

This repository provides the replication materials for the study:

**“The Impact of U.S. Monetary Policy Shocks on Thailand’s Domestic Value Added:  
A Local Projection and Production Network Analysis”**

The project examines how unexpected changes in U.S. monetary policy affect Thailand’s sectoral domestic value added (DVA) over 1998–2022, using high-frequency monetary policy surprises, OECD TiVA data, BIS real effective exchange rate data, and input-output network indicators.

---

## Repository Overview

This repository contains the code, notebook, and figure assets required to reproduce the Thai-language methodology notebook:

`chapter3_methodology_revised_TH.ipynb`

The repository is designed as a reproducible research workflow. Raw data files are not version-controlled because some inputs are large, externally licensed, or must be manually downloaded from the original data providers.

Generated datasets, intermediate outputs, and model results can be recreated by running the pipeline scripts.

---

## 1. What's in this Repo

```text
.
├── chapter3_methodology_revised_TH.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── process_fed_shocks.py
├── prepare_reer_controls.py
├── run_lp_reer_response.py
├── run_lp_sector_interactions.py
├── run_lp_analysis.py
├── local_projections_thailand.py
├── fed_shocks_outputs_readme.md
│
└── fed_dva_analysis/
    ├── scripts/
    │   ├── extract_oecd_linkages.py
    │   ├── create_sector_content_shares.py
    │   ├── sectoral_local_projections.py
    │   └── master_sector_analysis.py
    └── outputs/
        └── figures/

---

## 2. Quick Start

```bash
# Clone and enter the repo
git clone https://github.com/pearpcn/dva_response_fed_surprise.git
cd dva_response_fed_surprise

# Create environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# (Optional but recommended) Run the pipeline once to materialise all data.
# This takes a few minutes and downloads ~280 MB from BIS on first run.
python process_fed_shocks.py
python prepare_reer_controls.py
python run_lp_reer_response.py
python run_lp_sector_interactions.py

# Open the notebook
jupyter lab chapter3_methodology_revised_TH.ipynb
```

All file paths in the notebook are relative to the repo root, so no path edits
are required.

---

## 3. End-to-End Reproduction Pipeline

The repo ships **only code and figures**. To reproduce the data and re-run the
analysis from scratch:

| Step | Script | Inputs | Outputs |
|---:|---|---|---|
| 1 | `process_fed_shocks.py` | `data_raw/mps_SFFED.csv` (manual) + auto-fetch S&P 500 | `fed_shocks_{main_1998_2022,appendix_1995_1997,cleaned_annual}.csv`, `fed_shocks_outputs_readme.md` |
| 2 | `prepare_reer_controls.py` | `data_raw/{eer.csv (auto-DL), sectoral_dva_panel.csv, controls_annual.csv}` + `fed_shocks_main_1998_2022.csv` | `data_raw/reer_*.csv`, `data_raw/macro_controls_with_lags_annual.csv`, `fed_dva_analysis/outputs/panel_main_with_reer_controls.csv` |
| 3a | `run_lp_reer_response.py` | `panel_main_with_reer_controls.csv` | `figures/reer_lp_irf.png`, `results/reer_lp_results.csv` |
| 3b | `fed_dva_analysis/scripts/extract_oecd_linkages.py` | `data_raw/OECD_ICIO_2022.xlsx` (manual) | `outputs/linkages_oecd_thailand.csv` |
| 3c | `fed_dva_analysis/scripts/create_sector_content_shares.py` | `data_raw/IOTs_DOMIMP/THA2022dom.csv` (manual) | `outputs/sector_content_shares_2022.csv` |
| 3d | `fed_dva_analysis/scripts/sectoral_local_projections.py` | panel + linkages | `outputs/sectoral_significant_h0.csv` |
| 4 | `fed_dva_analysis/scripts/master_sector_analysis.py` | outputs of 3b–3d | `outputs/sector_master_table.csv` |
| 5 | `run_lp_sector_interactions.py` | panel + sector master + appendix shocks | `figures/irf_interactions_high_low_main.png`, `results/sector_lp_{interactions_main_comparison,horserace_main}.csv` |
| 6 | Notebook cells (Cells 23, 26, 31) | LP results + master table | `fig4_6_valuation_comparison.png`, `fig4_7_robustness_irf.png`, `figA1_correlation_heatmap.png` |

Sanity check built into the pipeline: `process_fed_shocks.py` asserts that the
main sample covers exactly **1998–2022** and the appendix covers exactly
**1995–1997**, and raises an `AssertionError` if either window is misaligned.

---

## 4. Required External Data (Manual Download)

Some raw inputs are too large or licence-restricted to redistribute. Place them
in `data_raw/` before running the pipeline:

| File | Source | Notes |
|---|---|---|
| `data_raw/mps_SFFED.csv` | [Bauer & Swanson (2023) replication archive](https://www.michaeldbauer.com/) | High-frequency FOMC monetary policy surprises |
| `data_raw/OECD_ICIO_2022.xlsx` | [OECD ICIO 2022 release](https://www.oecd.org/sti/ind/inter-country-input-output-tables.htm) | Inter-country I-O tables |
| `data_raw/IOTs_DOMIMP/THA2022dom.csv` | OECD ICIO (DOMIMP slice) | Thailand 2022 domestic I-O |
| `data_raw/sectoral_dva_panel.csv` | Constructed from OECD TiVA | Sectoral DVA panel (1995–2022) |
| `data_raw/controls_annual.csv` | World Bank / IMF / FRED | Annual macro controls |

`data_raw/eer.csv` (BIS broad real effective exchange rate, ~278 MB) is
**auto-downloaded** by `prepare_reer_controls.py` from the BIS bulk data portal
on first run.

---

## 5. Methodology Notes

- **Shock identification** follows Bauer & Swanson (2023) for the raw FOMC
  surprises and Jarociński & Karadi (2020)'s sign restriction for separating
  pure monetary policy shocks from the information channel.
- **REER source** is the BIS broad real effective exchange rate. The
  preparation script can either read a local `eer.csv` or auto-download from
  the BIS bulk data portal.
- **Estimation** uses panel local projections (Jordà 2005) with sector fixed
  effects and cluster-robust standard errors clustered by sector. The REER LP
  uses Newey-West HAC standard errors instead.
- The notebook is in **Thai**; variable names and code comments remain in
  English.

---

## 6. License & Citation

Add your preferred license (e.g. MIT, CC-BY) before publishing.
