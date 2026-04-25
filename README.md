# Chapter 3 — Methodology Replication Package (Thai Edition)

This repository contains the **complete code, data and figure assets** required
to reproduce `chapter3_methodology_revised_TH.ipynb` — the Thai-language
methodology chapter on the response of Thailand's domestic value-added (DVA) to
US Federal Reserve monetary policy shocks (1998–2022).

The folder structure mirrors the original working layout, so the notebook runs
without modification once the dependencies are installed.

---

## 1. Folder Structure

```
chapter3-methodology-TH-replication/
├── chapter3_methodology_revised_TH.ipynb     # The notebook (entry point)
├── requirements.txt                          # Python dependencies
├── README.md                                 # This file
│
├── process_fed_shocks.py                     # Step 1 — Fed shock processing
├── prepare_reer_controls.py                  # Step 2 — REER + macro controls
├── run_lp_reer_response.py                   # Step 3a — LP for REER response
├── run_lp_sector_interactions.py             # Step 3b — sector-level LP
├── run_lp_analysis.py                        # Wrapper / orchestrator
├── local_projections_thailand.py             # Core LP estimator class
│
├── fed_shocks_main_1998_2022.csv             # Output of Step 1 (main sample)
├── fed_shocks_appendix_1995_1997.csv         # Output of Step 1 (appendix)
├── fed_shocks_cleaned_annual.csv             # Output of Step 1 (full)
├── fed_shocks_outputs_readme.md              # Data dictionary for Step 1 outputs
│
├── data_raw/                                 # Raw / intermediate input data
│   ├── mps_SFFED.csv                         # Bauer–Swanson FOMC ME shocks
│   ├── eer.csv                               # BIS effective exchange rate
│   ├── sectoral_dva_panel.csv                # Sectoral DVA panel (1995–2022)
│   ├── controls_annual.csv                   # Annual macro controls
│   ├── ME_annual.csv                         # Pre-aggregated ME (intermediate)
│   ├── linkages_adb.csv                      # ADB I-O linkages (backup)
│   ├── OECD_ICIO_2022.xlsx                   # OECD ICIO 2022
│   ├── reer_th_broad_real_monthly.csv        # Output of Step 2 (intermediate)
│   ├── reer_th_broad_real_annual.csv         # Output of Step 2 (intermediate)
│   ├── macro_controls_with_lags_annual.csv   # Output of Step 2 (intermediate)
│   └── IOTs_DOMIMP/
│       └── THA2022dom.csv                    # Thailand 2022 I-O (DOMIMP)
│
└── fed_dva_analysis/
    ├── scripts/                              # Sector-level pipeline
    │   ├── extract_oecd_linkages.py
    │   ├── create_sector_content_shares.py
    │   ├── sectoral_local_projections.py
    │   └── master_sector_analysis.py
    └── outputs/
        ├── panel_main_with_reer_controls.csv      # Final estimation panel
        ├── linkages_oecd_thailand.csv             # OECD-derived linkages
        ├── sector_content_shares_2022.csv         # I-O content shares
        ├── sectoral_significant_h0.csv            # Sector-level significance (h=0)
        ├── sector_master_table.csv                # Master sector table
        ├── results/
        │   ├── reer_lp_results.csv
        │   ├── sector_lp_interactions_main_comparison.csv
        │   └── sector_lp_horserace_main.csv
        └── figures/                                # All 14 figures used in the chapter
            ├── reer_lp_irf.png
            ├── irf_interactions_high_low_main.png
            ├── horse_race_coefficients_clean.png
            ├── fig3_2_baseline_irf.png            # Fig. 4.1
            ├── fig4_2a_usd_vs_thb_panel_lp.png    # Fig. 4.2a
            ├── fig4_2b_fx_share.png               # Fig. 4.2b
            ├── fig4_2c_top_bottom5.png            # Fig. 4.2c
            ├── fig3_4_highlow_irf.png             # Fig. 4.3
            ├── fig3_5_horserace.png               # Fig. 4.4
            ├── fig3_6_reer_irf.png                # Fig. 4.5
            ├── fig4_6_valuation_comparison.png    # Fig. 4.6
            ├── fig4_7_robustness_irf.png          # Fig. 4.7
            ├── figA1_correlation_heatmap.png      # Fig. ก.1
            └── figA3_me_shock_appendix.png        # Fig. ก.3
```

---

## 2. Quick Start

```bash
# 1. Create environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Open the notebook
jupyter lab chapter3_methodology_revised_TH.ipynb
```

All file paths in the notebook are relative to this folder, so no path edits
are required.

---

## 3. End-to-End Reproduction Pipeline

The package is shipped with all intermediate and final outputs already
generated, so the notebook can be opened and inspected immediately. To
**re-run** the full pipeline from raw inputs:

| Step | Script | Inputs | Outputs |
|---:|---|---|---|
| 1 | `process_fed_shocks.py` | `data_raw/mps_SFFED.csv` (+ auto-fetch S&P 500) | `fed_shocks_{main,appendix,cleaned}*.csv`, `fed_shocks_outputs_readme.md` |
| 2 | `prepare_reer_controls.py` | `data_raw/{eer.csv, sectoral_dva_panel.csv, controls_annual.csv}` + `fed_shocks_main_1998_2022.csv` | `data_raw/reer_*.csv`, `data_raw/macro_controls_with_lags_annual.csv`, `fed_dva_analysis/outputs/panel_main_with_reer_controls.csv` |
| 3a | `run_lp_reer_response.py` | `panel_main_with_reer_controls.csv` | `figures/reer_lp_irf.png`, `results/reer_lp_results.csv` |
| 3b | `fed_dva_analysis/scripts/extract_oecd_linkages.py` | `data_raw/OECD_ICIO_2022.xlsx` | `outputs/linkages_oecd_thailand.csv` |
| 3c | `fed_dva_analysis/scripts/create_sector_content_shares.py` | `data_raw/IOTs_DOMIMP/THA2022dom.csv` | `outputs/sector_content_shares_2022.csv` |
| 3d | `fed_dva_analysis/scripts/sectoral_local_projections.py` | panel + linkages | `outputs/sectoral_significant_h0.csv` |
| 4 | `fed_dva_analysis/scripts/master_sector_analysis.py` | outputs of 3b–3d | `outputs/sector_master_table.csv` |
| 5 | `run_lp_sector_interactions.py` | panel + sector master + appendix shocks | `figures/irf_interactions_high_low_main.png`, `results/sector_lp_{interactions_main_comparison,horserace_main}.csv` |
| 6 | Notebook cells (Cells 23, 26, 31) | LP results + master table | `fig4_6_valuation_comparison.png`, `fig4_7_robustness_irf.png`, `figA1_correlation_heatmap.png` |

Sample sanity check: `process_fed_shocks.py` validates that the main sample
covers exactly **1998–2022** and the appendix covers exactly **1995–1997**;
it raises an `AssertionError` if either window is mis-aligned.

---

## 4. Notes

- **Shock identification** follows Bauer & Swanson (2023) for the raw FOMC
  surprises and Jarociński & Karadi (2020)'s sign restriction for separating
  pure monetary policy shocks from the information channel.
- **REER source** is the BIS broad real effective exchange rate. The
  preparation script can either read a local `eer.csv` or auto-download from
  the BIS bulk data portal.
- **Estimation** uses panel local projections (Jordà 2005) with sector fixed
  effects and cluster-robust standard errors clustered by sector.
- The notebook is in **Thai**; variable names and code comments remain in
  English.

---

## 5. License & Citation

Add your preferred license (e.g. MIT, CC-BY) before publishing.
