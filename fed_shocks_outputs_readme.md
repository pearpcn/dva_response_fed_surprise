# Fed Shock Outputs: Data Dictionary

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
