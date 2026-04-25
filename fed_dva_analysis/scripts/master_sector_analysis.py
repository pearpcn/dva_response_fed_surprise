"""
master_sector_analysis.py
=========================
Merges:
  1. sectoral_significant_h0.csv  — IRF coefficient at h=0
  2. linkages_oecd_thailand.csv   — BL / FL / Quadrant
  3. sector_content_shares_2022.csv — domestic / import / export content

Then answers: do high-impact (large positive IRF) sectors have
  - high domestic content  (keep income inside Thailand)
  - low import content     (less FX cost leakage)
  - identifiable network positions (Quadrant)?

Outputs
-------
  outputs/sector_master_table.csv
  outputs/fig_quadrant_vs_irf.png       BL–FL scatter, bubble = |IRF|, colour = direction
  outputs/fig_content_vs_irf.png        domestic vs import share, bubble = |IRF|
  outputs/fig_content_group_compare.png bar chart: mean IRF by content group
  outputs/fig_bwfl_content_scatter.png  bwd_norm vs domestic content, colour = IRF direction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings, os

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(BASE, "outputs")

# ─────────────────────────────────────────
# 1. Load
# ─────────────────────────────────────────
irf = pd.read_csv(os.path.join(OUT, "sectoral_significant_h0.csv"))
lnk = pd.read_csv(os.path.join(OUT, "linkages_oecd_thailand.csv"))
cnt = pd.read_csv(os.path.join(OUT, "sector_content_shares_2022.csv"))

# keep only OECD sectors from IRF; drop aggregates not in 50-sector list
valid_50 = set(lnk["sector"])
irf_oecd = irf[irf["source"] == "OECD"].copy()
irf_50   = irf_oecd[irf_oecd["sector"].isin(valid_50)].copy()

# ─────────────────────────────────────────
# 2. Merge
# ─────────────────────────────────────────
# base: all 50 structural sectors
base = lnk.merge(cnt[["sector","domestic_content_share","import_content_share",
                        "export_content_share","total_intermediate_share"]],
                 on="sector", how="left")

# add IRF (some sectors may not be significant → NaN)
base = base.merge(irf_50[["sector","coef","stderr","pval","sig","direction"]],
                  on="sector", how="left")

# ─────────────────────────────────────────
# 3. Quadrant classification
# ─────────────────────────────────────────
def quadrant(row):
    bw = row["bwd_norm"] >= 1.0
    fw = row["fwd_norm"] >= 1.0
    if bw and fw:  return "Q1 Key (High BL+FL)"
    if fw and not bw: return "Q2 Forward (Low BL, High FL)"
    if bw and not fw: return "Q4 Backward (High BL, Low FL)"
    return "Q3 Weak (Low BL+FL)"

base["quadrant"] = base.apply(quadrant, axis=1)

# ─────────────────────────────────────────
# 4. Content-group classification
# ─────────────────────────────────────────
def content_group(row):
    d = row["domestic_content_share"]
    m = row["import_content_share"]
    e = row["export_content_share"]
    dom = max(d, m, e)
    if dom == d and d >= 0.40:
        return "Domestic-intensive"
    if dom == m or m >= 0.25:
        return "Import-intensive"
    if dom == e or e >= 0.25:
        return "Export-intensive"
    return "Domestic-intensive"   # default

base["content_group"] = base.apply(content_group, axis=1)

# ─────────────────────────────────────────
# 5. Save master table
# ─────────────────────────────────────────
cols_order = ["sector","quadrant","content_group",
              "bwd_norm","fwd_norm","va_multiplier_simple",
              "domestic_content_share","import_content_share","export_content_share",
              "total_intermediate_share",
              "coef","stderr","pval","sig","direction"]
base[cols_order].to_csv(os.path.join(OUT, "sector_master_table.csv"), index=False)
print(f"Saved sector_master_table.csv  ({len(base)} rows)")

# helper: significant only
sig = base[base["coef"].notna()].copy()
sig["abs_coef"] = sig["coef"].abs()
print(f"  ↳ {len(sig)} sectors with significant IRF at h=0")

# ─────────────────────────────────────────
# COLOUR MAPS
# ─────────────────────────────────────────
Q_COLORS = {"Q1 Key (High BL+FL)":        "#d62728",
            "Q2 Forward (Low BL, High FL)":"#ff7f0e",
            "Q3 Weak (Low BL+FL)":         "#aec7e8",
            "Q4 Backward (High BL, Low FL)":"#1f77b4"}

CG_COLORS = {"Domestic-intensive": "#2ca02c",
             "Import-intensive":   "#d62728",
             "Export-intensive":   "#ff7f0e"}

DIR_COLORS = {"Positive": "#2ca02c", "Negative": "#d62728"}

# ─────────────────────────────────────────
# FIG 1: BL–FL Quadrant scatter, bubble = |IRF|, colour = quadrant
# ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))

for q, grp in base.groupby("quadrant"):
    has_irf = grp[grp["coef"].notna()]
    no_irf  = grp[grp["coef"].isna()]
    c = Q_COLORS[q]

    # no IRF: small hollow circle
    ax.scatter(no_irf["bwd_norm"], no_irf["fwd_norm"],
               s=40, edgecolors=c, facecolors="none", linewidths=0.8, alpha=0.5)
    # has IRF: filled, size ∝ |coef|
    if len(has_irf):
        ax.scatter(has_irf["bwd_norm"], has_irf["fwd_norm"],
                   s=has_irf["coef"].abs() * 1200, color=c, alpha=0.75, label=q)

# label significant sectors
for _, r in sig.iterrows():
    ax.annotate(r["sector"], (r["bwd_norm"], r["fwd_norm"]),
                fontsize=7, ha="left", va="bottom",
                xytext=(3,3), textcoords="offset points")

ax.axvline(1, color="grey", lw=0.8, ls="--")
ax.axhline(1, color="grey", lw=0.8, ls="--")
ax.set_xlabel("Backward Linkage (normalised)", fontsize=12)
ax.set_ylabel("Forward Linkage (normalised)", fontsize=12)
ax.set_title("Production Network Quadrant vs IRF at h=0\n"
             "(bubble size = |coefficient|, filled = significant)", fontsize=13)
ax.legend(title="Quadrant", loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_quadrant_vs_irf.png"), dpi=180)
plt.close()
print("Saved fig_quadrant_vs_irf.png")

# ─────────────────────────────────────────
# FIG 2: Domestic content vs Import content
#         bubble = |IRF|, colour = direction, only significant
# ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

for d, grp in sig.groupby("direction"):
    c = DIR_COLORS.get(d, "grey")
    sc = ax.scatter(grp["import_content_share"], grp["domestic_content_share"],
                    s=grp["abs_coef"] * 1500, color=c, alpha=0.7,
                    edgecolors="white", linewidths=0.5, label=d)

for _, r in sig.iterrows():
    ax.annotate(r["sector"], (r["import_content_share"], r["domestic_content_share"]),
                fontsize=7.5, ha="left", va="bottom",
                xytext=(4, 2), textcoords="offset points")

ax.axvline(0.25, color="grey", lw=0.8, ls="--", label="Import threshold 25%")
ax.axhline(0.40, color="steelblue", lw=0.8, ls="--", label="Domestic threshold 40%")
ax.set_xlabel("Import Content Share (M / Output)", fontsize=12)
ax.set_ylabel("Domestic Content Share (VA / Output)", fontsize=12)
ax.set_title("Content Structure vs IRF Direction at h=0\n"
             "(bubble = |coef|, green=Positive, red=Negative)", fontsize=13)
ax.legend(fontsize=9)
# annotate zone labels
ax.text(0.02, 0.85, "Domestic-\nintensive\n✓ benefits", transform=ax.transAxes,
        color="darkgreen", fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="honeydew", ec="darkgreen", lw=0.6))
ax.text(0.68, 0.08, "Import-\nintensive\n⚠ FX cost", transform=ax.transAxes,
        color="darkred", fontsize=9, va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", ec="darkred", lw=0.6))
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_content_vs_irf.png"), dpi=180)
plt.close()
print("Saved fig_content_vs_irf.png")

# ─────────────────────────────────────────
# FIG 3: Mean IRF by content group — bar chart
# ─────────────────────────────────────────
cg_stats = (sig.groupby("content_group")["coef"]
              .agg(["mean","sem","count"])
              .rename(columns={"mean":"mean_coef","sem":"se","count":"n"})
              .reset_index())

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(cg_stats["content_group"], cg_stats["mean_coef"],
              color=[CG_COLORS[g] for g in cg_stats["content_group"]],
              alpha=0.8, edgecolor="white", width=0.5)
ax.errorbar(cg_stats["content_group"], cg_stats["mean_coef"],
            yerr=cg_stats["se"] * 1.96, fmt="none", color="black", capsize=5, lw=1.5)

for bar, (_, row) in zip(bars, cg_stats.iterrows()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"n={int(row['n'])}\nβ={row['mean_coef']:.3f}",
            ha="center", va="bottom", fontsize=9)

ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Mean IRF Coefficient at h=0", fontsize=12)
ax.set_title("Mean DVA Response to Fed Shock by Content Group\n(significant sectors only, 95% CI)", fontsize=12)
ax.set_ylim(bottom=min(0, cg_stats["mean_coef"].min() - 0.1))
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_content_group_compare.png"), dpi=180)
plt.close()
print("Saved fig_content_group_compare.png")

# ─────────────────────────────────────────
# FIG 4: BL (bwd_norm) vs Domestic content share
#         colour = IRF direction, shape = quadrant, ALL 50 sectors
# ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

markers = {"Q1 Key (High BL+FL)": "o",
           "Q2 Forward (Low BL, High FL)": "^",
           "Q3 Weak (Low BL+FL)": "s",
           "Q4 Backward (High BL, Low FL)": "D"}

for q, qgrp in base.groupby("quadrant"):
    for d, grp in qgrp.groupby("direction", dropna=False):
        has = grp[grp["direction"].notna()]
        no  = grp[grp["direction"].isna()]
        m   = markers[q]
        c   = DIR_COLORS.get(d, "#aaaaaa")
        ax.scatter(no["bwd_norm"], no["domestic_content_share"],
                   s=35, marker=m, edgecolors="#aaaaaa", facecolors="none",
                   linewidths=0.7, alpha=0.5)
        if len(has):
            ax.scatter(has["bwd_norm"], has["domestic_content_share"],
                       s=has["coef"].abs() * 1200 + 40, marker=m,
                       color=c, alpha=0.75, edgecolors="white", linewidths=0.4)

for _, r in sig.iterrows():
    ax.annotate(r["sector"], (r["bwd_norm"], r["domestic_content_share"]),
                fontsize=7, xytext=(3, 2), textcoords="offset points")

ax.axvline(1, color="grey", lw=0.8, ls="--")
ax.axhline(0.40, color="steelblue", lw=0.8, ls="--")
ax.set_xlabel("Backward Linkage Index (normalised)", fontsize=12)
ax.set_ylabel("Domestic Content Share (VA / Output)", fontsize=12)
ax.set_title("Backward Linkage vs Domestic Content Structure\n"
             "(shape=Quadrant | filled+sized=significant | green=positive response)", fontsize=12)

legend_elems = []
for q, m in markers.items():
    legend_elems.append(plt.scatter([],[], marker=m, color="grey", s=50, label=q))
legend_elems.append(mpatches.Patch(color="#2ca02c", label="Positive IRF"))
legend_elems.append(mpatches.Patch(color="#d62728", label="Negative IRF"))
ax.legend(handles=legend_elems, fontsize=8, loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_bwfl_content_scatter.png"), dpi=180)
plt.close()
print("Saved fig_bwfl_content_scatter.png")

# ─────────────────────────────────────────
# 6. Consistency check summary
# ─────────────────────────────────────────
print("\n══════════════════════════════════")
print("CONSISTENCY CHECK: Quadrant × IRF direction")
print("══════════════════════════════════")
ct = (sig.groupby(["quadrant","direction"])
         .size().unstack(fill_value=0)
         .reset_index())
print(ct.to_string(index=False))

print("\n══════════════════════════════════")
print("CONSISTENCY CHECK: Content Group × IRF direction")
print("══════════════════════════════════")
ct2 = (sig.groupby(["content_group","direction"])
          .size().unstack(fill_value=0)
          .reset_index())
print(ct2.to_string(index=False))

print("\n══════════════════════════════════")
print("Content Group × Quadrant (significant sectors)")
print("══════════════════════════════════")
ct3 = pd.crosstab(sig["quadrant"], sig["content_group"])
print(ct3.to_string())

print("\n══════════════════════════════════")
print("Top 10 by |IRF coef| with context")
print("══════════════════════════════════")
top10 = sig.nlargest(10, "abs_coef")[
    ["sector","coef","direction","quadrant","content_group",
     "domestic_content_share","import_content_share","export_content_share"]
].round(4)
print(top10.to_string(index=False))

print("\nDone.")
