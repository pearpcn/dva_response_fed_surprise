"""
OECD ICIO 2022 - Thailand Linkage Analysis
===========================================
Extract Thailand's domestic IO matrix and compute linkage indicators
comparable to ADB MRIO linkages

Author: [Thesis Project]
Date: 2026-02-09
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================
WORKSPACE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = WORKSPACE_DIR / 'data_raw'
OUTPUT_DIR = WORKSPACE_DIR / 'fed_dva_analysis' / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ICIO_FILE = DATA_DIR / 'OECD_ICIO_2022.xlsx'
COUNTRY_CODE = 'THA'  # Thailand

print("=" * 80)
print("OECD ICIO 2022 - THAILAND LINKAGE ANALYSIS")
print("=" * 80)
print()

# ==============================================================================
# 1. Load and Inspect ICIO Structure
# ==============================================================================
print("📂 Step 1: Loading and inspecting OECD ICIO 2022 structure...")
print()

if not ICIO_FILE.exists():
    print(f"   ✗ ERROR: File not found: {ICIO_FILE}")
    print("   Please ensure OECD_ICIO_2022.xlsx is in data_raw/ directory")
    exit(1)

# Load Excel file
print(f"   Loading: {ICIO_FILE.name} (this may take a moment...)")
xls = pd.ExcelFile(ICIO_FILE, engine='openpyxl')

print(f"   ✓ File loaded successfully")
print(f"   Sheet names: {xls.sheet_names}")
print()

# Use the first sheet (typically the IO matrix)
sheet_name = xls.sheet_names[0]
print(f"   Using sheet: '{sheet_name}'")
print()

# Load with row/column headers
print("   Reading data (this will take some time for large files)...")
df = pd.read_excel(xls, sheet_name=sheet_name, header=None)

print(f"   ✓ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print()

# Inspect structure
print("   First 10 rows and 10 columns:")
print(df.iloc[:10, :10])
print()

# ==============================================================================
# 2. Parse ICIO Structure
# ==============================================================================
print("=" * 80)
print("📊 Step 2: Parsing ICIO structure...")
print()

# OECD ICIO format:
# - Row 1: Column headers (country_sector format, e.g., AUS_A01, THA_C26)
# - Column 1: Row headers (country_sector format)
# - Data starts at (2, 2)

print("   Detecting header structure...")

# Extract headers from row 1 and column 1
col_headers = df.iloc[1, :].tolist()
row_headers = df.iloc[:, 1].tolist()

print(f"   ✓ Column headers: {len(col_headers)}")
print(f"   ✓ Row headers: {len(row_headers)}")
print()

# Extract data matrix (skip first 2 rows and first 2 columns)
Z = df.iloc[2:, 2:].values.astype(float)
print(f"   ✓ Transaction matrix Z: {Z.shape}")
print()

# Parse country codes from headers (format: COUNTRY_SECTOR)
def parse_country_sector(header):
    """Extract country and sector from header string"""
    if isinstance(header, str) and '_' in header:
        parts = header.split('_', 1)
        return parts[0], parts[1] if len(parts) > 1 else ''
    return None, None

# Extract countries from column headers
col_countries, col_sectors = zip(*[parse_country_sector(h) for h in col_headers[2:]])
row_countries_parsed, row_sectors = zip(*[parse_country_sector(h) for h in row_headers[2:]])

# Show unique countries
unique_countries = sorted(set([c for c in col_countries if c]))
print(f"   Countries in dataset: {len(unique_countries)}")
print(f"   Sample: {unique_countries[:15]}")
print()

if COUNTRY_CODE not in unique_countries:
    print(f"   ✗ ERROR: Country code '{COUNTRY_CODE}' not found in dataset")
    print(f"   Available codes: {unique_countries}")
    exit(1)

# ==============================================================================
# 3. Extract Thailand Domestic IO Matrix
# ==============================================================================
print("=" * 80)
print("🇹🇭 Step 3: Extracting Thailand domestic IO matrix...")
print()

# Find Thailand indices in both rows and columns
tha_row_idx = [i for i, c in enumerate(row_countries_parsed) if c == COUNTRY_CODE]
tha_col_idx = [i for i, c in enumerate(col_countries) if c == COUNTRY_CODE]

print(f"   Thailand rows: {len(tha_row_idx)} sectors")
print(f"   Thailand columns: {len(tha_col_idx)} sectors")
print()

if len(tha_row_idx) == 0 or len(tha_col_idx) == 0:
    print("   ✗ ERROR: No Thailand data found")
    exit(1)

# Extract Value-Added row (row 803 in OECD ICIO)
print("   Extracting value-added data...")
try:
    # VA is at row 803 (absolute row in Excel), columns are THA sectors
    # Add 2 to tha_col_idx to account for first 2 columns
    va_absolute_cols = [i + 2 for i in tha_col_idx]
    va_data = df.iloc[803, va_absolute_cols].values.astype(float)
    print(f"   ✓ Value-added extracted: {len(va_data)} sectors")
    print(f"      Mean VA: {va_data.mean():.2f}")
    print(f"      Range: [{va_data.min():.2f}, {va_data.max():.2f}]")
except Exception as e:
    print(f"   ⚠ Warning: Could not extract VA data: {e}")
    print(f"   Will use approximation: v_i = 1 - Σa_ij")
    va_data = None

# Extract Total Output row (row 804 in OECD ICIO)
print("   Extracting total output data...")
try:
    x_absolute_cols = [i + 2 for i in tha_col_idx]
    x_data_raw = df.iloc[804, x_absolute_cols]
    # Check if data is valid
    if x_data_raw.isna().all() or (x_data_raw == 0).all():
        print(f"   ⚠ Row 804 has no valid data, will compute from Z matrix")
        x_data = None
    else:
        x_data = x_data_raw.values.astype(float)
        print(f"   ✓ Total output extracted: {len(x_data)} sectors")
        print(f"      Mean output: {x_data.mean():.2f}")
        print(f"      Range: [{x_data.min():.2f}, {x_data.max():.2f}]")
except Exception as e:
    print(f"   ⚠ Warning: Could not extract output data: {e}")
    x_data = None
print()

# Extract Thailand sectors (from row headers)
tha_sectors = np.array([row_sectors[i] for i in tha_row_idx])

print(f"   Thailand sectors:")
for i, sector in enumerate(tha_sectors[:20]):  # Show first 20
    print(f"      {i+1:2d}. {sector}")
if len(tha_sectors) > 20:
    print(f"      ... and {len(tha_sectors)-20} more")
print()

# Extract domestic use matrix Z_TH
# Need to ensure we get square matrix - use minimum dimension
n_sectors = min(len(tha_row_idx), len(tha_col_idx))

# Use first n_sectors from both
Z_TH = Z[np.ix_(tha_row_idx[:n_sectors], tha_col_idx[:n_sectors])]
tha_sectors = np.array([row_sectors[i] for i in tha_row_idx[:n_sectors]])

# Adjust va_data and x_data to match n_sectors
if va_data is not None:
    va_data = va_data[:n_sectors]
if x_data is not None:
    x_data = x_data[:n_sectors]

print(f"   ✓ Domestic use matrix Z_TH: {Z_TH.shape} (square)")
print()

# Extract output vector
# Output is sum of all uses across all countries (full row sum)
x_TH = Z[tha_row_idx[:n_sectors], :].sum(axis=1)  # Total output = sum of all column uses

# If we have x_data from row 804, use it if valid
if x_data is not None and not np.isnan(x_data).any():
    print(f"   ✓ Using total output from row 804: {len(x_data)} sectors")
    x_TH = x_data
else:
    print(f"   ✓ Computed output vector from Z matrix: {len(x_TH)} sectors")

# Use x_TH for VA coefficient calculation
if va_data is not None:
    # Use x_TH (computed output) for VA coefficients
    print(f"   Using x_TH for VA coefficient calculation")
    x_for_va = x_TH
else:
    x_for_va = x_TH

print(f"   ✓ Output vector x_TH: {len(x_TH)} sectors")
print()

# Check for zero or negative outputs
zero_output = (x_TH <= 0)
if zero_output.any():
    print(f"   ⚠ Warning: {zero_output.sum()} sectors have zero or negative output")
    print(f"   Removing these sectors...")
    
    valid_idx = ~zero_output
    Z_TH = Z_TH[np.ix_(valid_idx, valid_idx)]
    x_TH = x_TH[valid_idx]
    tha_sectors = tha_sectors[valid_idx]
    
    print(f"   ✓ Valid sectors remaining: {len(tha_sectors)}")
    print()

# Summary statistics
print("   Summary statistics:")
print(f"      Total output sum: {x_TH.sum():,.0f}")
print(f"      Mean output: {x_TH.mean():,.0f}")
print(f"      Output range: [{x_TH.min():,.0f}, {x_TH.max():,.0f}]")
print()

# ==============================================================================
# 4. Compute Technical Coefficients and Leontief Inverse
# ==============================================================================
print("=" * 80)
print("🔢 Step 4: Computing technical coefficients and Leontief inverse...")
print()

# Technical coefficient matrix A = Z / x (column-wise division)
# A[i,j] = Z[i,j] / x[j]  (input from i needed per unit output of j)
print("   Computing technical coefficient matrix A...")

x_diag_inv = np.diag(1.0 / x_TH)
A_TH = Z_TH @ x_diag_inv

print(f"   ✓ Technical coefficient matrix A_TH: {A_TH.shape}")
print(f"      Mean coefficient: {A_TH.mean():.4f}")
print(f"      Max coefficient: {A_TH.max():.4f}")
print()

# Leontief inverse L = (I - A)^(-1)
print("   Computing Leontief inverse L = (I - A)^(-1)...")

n = len(tha_sectors)
I = np.eye(n)

try:
    L_TH = np.linalg.inv(I - A_TH)
    print(f"   ✓ Leontief inverse computed successfully")
except np.linalg.LinAlgError:
    print(f"   ⚠ Singular matrix, using pseudo-inverse...")
    L_TH = np.linalg.pinv(I - A_TH)

print(f"      L_TH shape: {L_TH.shape}")
print(f"      Mean multiplier: {L_TH.mean():.4f}")
print()

# ==============================================================================
# 5. Compute Linkage Indicators
# ==============================================================================
print("=" * 80)
print("🔗 Step 5: Computing linkage indicators...")
print()

# Backward linkage (BL): sum of column j in A (total inputs per unit output)
BL = A_TH.sum(axis=0)  # Sum down columns
print(f"   ✓ Backward linkages computed: {len(BL)} sectors")
print(f"      Mean: {BL.mean():.4f}")
print(f"      Range: [{BL.min():.4f}, {BL.max():.4f}]")
print()

# Forward linkage (FL): sum of row i in A (total outputs per unit output)
FL = A_TH.sum(axis=1)  # Sum across rows
print(f"   ✓ Forward linkages computed: {len(FL)} sectors")
print(f"      Mean: {FL.mean():.4f}")
print(f"      Range: [{FL.min():.4f}, {FL.max():.4f}]")
print()

# Value-added coefficients v_i = VA_i / X_i
if va_data is not None and x_for_va is not None:
    # Use actual VA data from ICIO
    v = va_data / np.maximum(x_for_va, 1.0)  # Prevent division by zero
    v = np.maximum(v, 0.0)  # Ensure non-negative
    v = np.minimum(v, 1.0)  # Cap at 1.0 (cannot exceed 100%)
    print(f"   ✓ Value-added coefficients computed from ICIO data")
    print(f"      Mean VA coefficient: {v.mean():.4f}")
    print(f"      Range: [{v.min():.4f}, {v.max():.4f}]")
else:
    # Fallback: v_i = 1 - sum_j A[j,i] (residual after intermediate inputs)
    v = 1.0 - BL
    v = np.maximum(v, 0.01)  # Ensure positive (minimum 1%)
    print(f"   ⚠ Using approximation for VA coefficients")
    print(f"      Mean VA share: {v.mean():.4f}")
    print(f"      Range: [{v.min():.4f}, {v.max():.4f}]")
print()

# Value-added multiplier (simple)
# m_j = sum_i (v_i * L[i,j])  (total VA generated per unit final demand in sector j)
V = np.diag(v)
VA_multipliers_simple = (V @ L_TH).sum(axis=0)

print(f"   ✓ Value-added multipliers (simple) computed")
print(f"      Mean multiplier: {VA_multipliers_simple.mean():.4f}")
print(f"      Range: [{VA_multipliers_simple.min():.4f}, {VA_multipliers_simple.max():.4f}]")

# Value-added multiplier Type I
# Type I = sum of Leontief inverse column (total output multiplier)
VA_multipliers_typeI = L_TH.sum(axis=0)

print(f"   ✓ Value-added multipliers (Type I) computed")
print(f"      Mean Type I multiplier: {VA_multipliers_typeI.mean():.4f}")
print(f"      Range: [{VA_multipliers_typeI.min():.4f}, {VA_multipliers_typeI.max():.4f}]")
print()

# ==============================================================================
# 6. Normalize Linkages
# ==============================================================================
print("=" * 80)
print("📏 Step 6: Normalizing linkages...")
print()

# Normalize by mean
BL_mean = BL.mean()
FL_mean = FL.mean()

bwd_norm = BL / BL_mean
fwd_norm = FL / FL_mean

print(f"   Mean backward linkage: {BL_mean:.4f}")
print(f"   Mean forward linkage: {FL_mean:.4f}")
print()

print(f"   ✓ Normalized backward linkages")
print(f"      Mean: {bwd_norm.mean():.4f} (should be 1.0)")
print(f"      Range: [{bwd_norm.min():.4f}, {bwd_norm.max():.4f}]")
print()

print(f"   ✓ Normalized forward linkages")
print(f"      Mean: {fwd_norm.mean():.4f} (should be 1.0)")
print(f"      Range: [{fwd_norm.min():.4f}, {fwd_norm.max():.4f}]")
print()

# ==============================================================================
# 7. Create Results DataFrame
# ==============================================================================
print("=" * 80)
print("📋 Step 7: Creating results DataFrame...")
print()

results_df = pd.DataFrame({
    'sector': tha_sectors,
    'BL': BL,
    'FL': FL,
    'bwd_norm': bwd_norm,
    'fwd_norm': fwd_norm,
    'va_multiplier_simple': VA_multipliers_simple,
    'va_multiplier_typeI': VA_multipliers_typeI
})

print(f"   ✓ DataFrame created: {len(results_df)} sectors × {len(results_df.columns)} columns")
print()

# Show sample
print("   Sample results (first 10 sectors):")
print(results_df.head(10).to_string(index=False))
print()

# ==============================================================================
# 8. Export Results
# ==============================================================================
print("=" * 80)
print("💾 Step 8: Exporting results...")
print()

output_file = OUTPUT_DIR / 'linkages_oecd_thailand.csv'
results_df.to_csv(output_file, index=False)

print(f"   ✓ Saved: {output_file}")
print()

# ==============================================================================
# 9. Summary Statistics
# ==============================================================================
print("=" * 80)
print("📊 SUMMARY STATISTICS")
print("=" * 80)
print()

summary_stats = results_df[['bwd_norm', 'fwd_norm', 'va_multiplier_simple']].describe()
print(summary_stats)
print()

# ==============================================================================
# 10. Top Sectors
# ==============================================================================
print("=" * 80)
print("🏆 TOP 10 SECTORS BY LINKAGE TYPE")
print("=" * 80)
print()

print("Top 10 by Backward Linkage (Normalized):")
print("-" * 80)
top_bwd = results_df.nlargest(10, 'bwd_norm')[['sector', 'bwd_norm', 'BL']]
for idx, row in top_bwd.iterrows():
    print(f"   {row['sector']:15s} | Norm: {row['bwd_norm']:.3f} | Raw: {row['BL']:.3f}")
print()

print("Top 10 by Forward Linkage (Normalized):")
print("-" * 80)
top_fwd = results_df.nlargest(10, 'fwd_norm')[['sector', 'fwd_norm', 'FL']]
for idx, row in top_fwd.iterrows():
    print(f"   {row['sector']:15s} | Norm: {row['fwd_norm']:.3f} | Raw: {row['FL']:.3f}")
print()

print("Top 10 by Value-Added Multiplier (Simple):")
print("-" * 80)
top_va = results_df.nlargest(10, 'va_multiplier_simple')[['sector', 'va_multiplier_simple']]
for idx, row in top_va.iterrows():
    print(f"   {row['sector']:15s} | VA Mult (Simple): {row['va_multiplier_simple']:.3f}")
print()

print("Top 10 by Value-Added Multiplier (Type I):")
print("-" * 80)
top_va_typeI = results_df.nlargest(10, 'va_multiplier_typeI')[['sector', 'va_multiplier_typeI']]
for idx, row in top_va_typeI.iterrows():
    print(f"   {row['sector']:15s} | VA Mult (Type I): {row['va_multiplier_typeI']:.3f}")
print()

# ==============================================================================
# 11. Key Sectors Classification
# ==============================================================================
print("=" * 80)
print("🎯 KEY SECTORS CLASSIFICATION")
print("=" * 80)
print()

# Sectors with both high backward AND high forward linkages (>1.0)
key_sectors = results_df[(results_df['bwd_norm'] > 1.0) & (results_df['fwd_norm'] > 1.0)]

print(f"Key sectors (both BWD > 1.0 AND FWD > 1.0): {len(key_sectors)} sectors")
if len(key_sectors) > 0:
    print()
    for idx, row in key_sectors.iterrows():
        print(f"   {row['sector']:15s} | BWD: {row['bwd_norm']:.3f} | FWD: {row['fwd_norm']:.3f}")
    print()

# High backward only
high_bwd = results_df[(results_df['bwd_norm'] > 1.0) & (results_df['fwd_norm'] <= 1.0)]
print(f"High backward only: {len(high_bwd)} sectors")

# High forward only
high_fwd = results_df[(results_df['bwd_norm'] <= 1.0) & (results_df['fwd_norm'] > 1.0)]
print(f"High forward only: {len(high_fwd)} sectors")

# Low linkages
low_both = results_df[(results_df['bwd_norm'] <= 1.0) & (results_df['fwd_norm'] <= 1.0)]
print(f"Low linkages (both < 1.0): {len(low_both)} sectors")
print()

# ==============================================================================
# 12. Completion
# ==============================================================================
print("=" * 80)
print("✓ ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 80)
print()

print("Output files:")
print(f"   1. {output_file.name}")
print()

print("Next steps:")
print("   • Use this file to create network structure plots for OECD")
print("   • Compare with ADB linkages (linkages_adb.csv)")
print("   • Join with sectoral LP results via 'sector' column")
print()

print("Note: Sector codes in OECD ICIO may differ from ADB MRIO.")
print("      You may need to create a mapping file to harmonize sectors.")
