"""
Sectoral Local Projections Analysis
====================================
Analyze US Monetary Policy Shock impact on ALL Thai industrial sectors

Author: [Thesis Project]
Date: 2026-02-09
Method: Jordà (2005) Local Projections with Newey-West HAC standard errors
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.stats.sandwich_covariance import cov_hac
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

# ==============================================================================
# Configuration
# ==============================================================================
WORKSPACE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = WORKSPACE_DIR / 'main_thesis_final.xlsx'
OUTPUT_DIR = WORKSPACE_DIR / 'fed_dva_analysis' / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = list(range(7))  # h = 0, 1, 2, 3, 4, 5, 6
NEWEY_WEST_LAGS = 3
MIN_OBS = 10  # Minimum observations required for regression

print("=" * 80)
print("SECTORAL LOCAL PROJECTIONS ANALYSIS")
print("=" * 80)
print()

# ==============================================================================
# 1. Load Data
# ==============================================================================
print("📂 Loading data...")

df = pd.read_excel(DATA_FILE, sheet_name='Master_Data')
print(f"   ✓ Loaded: {len(df)} observations, {len(df.columns)} columns")
print(f"   Year range: {df['year'].min()}-{df['year'].max()}")
print()

# ==============================================================================
# 2. Identify Sector Columns Automatically
# ==============================================================================
print("🔍 Identifying sector columns...")

# Exclude non-sector columns
exclude_keywords = [
    'year', '_T', 'ln_', 'FX', 'PPI', 'GFC', 'ME_', 'dummy_',
    'factor', 'annual', 'shock'
]

def is_sector_column(col_name):
    """Check if column is a sector DVA column"""
    col_str = str(col_name)
    
    # Exclude if matches any keyword
    for keyword in exclude_keywords:
        if keyword in col_str:
            return False
    
    # Must be non-empty and not just whitespace
    if not col_str or col_str.isspace():
        return False
    
    return True

# Separate OECD and ADB sectors
oecd_sectors = []
adb_sectors = []

for col in df.columns:
    if is_sector_column(col):
        col_str = str(col)
        if col_str.startswith('adb_'):
            # ADB sector
            adb_sectors.append(col)
        else:
            # OECD sector
            oecd_sectors.append(col)

print(f"   ✓ OECD sectors identified: {len(oecd_sectors)}")
print(f"   ✓ ADB sectors identified: {len(adb_sectors)}")
print()

# Display sample sectors
print("Sample OECD sectors:", oecd_sectors[:10])
print("Sample ADB sectors:", adb_sectors[:10])
print()

# ==============================================================================
# 3. Create Crisis Dummies (if not exist)
# ==============================================================================
print("🏷️  Creating crisis dummy variables...")

if 'dummy_2008' not in df.columns:
    df['dummy_2008'] = (df['year'] == 2008).astype(int)
    print("   ✓ dummy_2008 created")

if 'dummy_2020' not in df.columns:
    df['dummy_2020'] = (df['year'] == 2020).astype(int)
    print("   ✓ dummy_2020 created")

print()

# ==============================================================================
# 4. Create Log-Transformed Sector Variables
# ==============================================================================
print("📊 Creating log-transformed sector variables...")

# Create log variables for all sectors
for sector in oecd_sectors + adb_sectors:
    log_col = f'ln_{sector}'
    if log_col not in df.columns:
        # Replace 0 and negative with NaN before log
        sector_data = pd.to_numeric(df[sector], errors='coerce')
        sector_data = sector_data.where(sector_data > 0, np.nan)
        df[log_col] = np.log(sector_data)

print(f"   ✓ Created log variables for {len(oecd_sectors) + len(adb_sectors)} sectors")
print()

# ==============================================================================
# 5. Define Control Variables
# ==============================================================================
control_vars = ['ln_FX', 'ln_PPI', 'GFC_factor', 'dummy_2008', 'dummy_2020']
shock_var = 'ME_shock_annual'

print("🔧 Model specification:")
print(f"   Shock variable: {shock_var}")
print(f"   Control variables: {', '.join(control_vars)}")
print(f"   Horizons: h = {min(HORIZONS)} to {max(HORIZONS)}")
print(f"   Standard errors: Newey-West (HAC) with {NEWEY_WEST_LAGS} lags")
print(f"   Minimum observations: {MIN_OBS}")
print()

# ==============================================================================
# 6. Function to Run LP for One Sector
# ==============================================================================
def run_local_projections_sector(df, sector_name, horizons, shock_var, control_vars):
    """Run Local Projections for a single sector"""
    
    results = {
        'sector': sector_name,
        'horizons': [],
        'coef': [],
        'stderr': [],
        'tstat': [],
        'pval': [],
        'ci_lower': [],
        'ci_upper': [],
        'n_obs': []
    }
    
    log_sector = f'ln_{sector_name}'
    
    # Check if log sector exists
    if log_sector not in df.columns:
        return None
    
    for h in horizons:
        try:
            # Create forward difference
            if h == 0:
                y = df[log_sector].diff()  # First difference
            else:
                y = df[log_sector].shift(-h) - df[log_sector]  # Cumulative change
            
            # Prepare X
            X_cols = [shock_var] + control_vars
            X = df[X_cols].copy()
            
            # Create clean dataset
            data = pd.concat([y, X], axis=1)
            data = data.dropna()
            
            # Check minimum observations
            if len(data) < MIN_OBS:
                results['horizons'].append(h)
                results['coef'].append(np.nan)
                results['stderr'].append(np.nan)
                results['tstat'].append(np.nan)
                results['pval'].append(np.nan)
                results['ci_lower'].append(np.nan)
                results['ci_upper'].append(np.nan)
                results['n_obs'].append(len(data))
                continue
            
            y_clean = data.iloc[:, 0]
            X_clean = data.iloc[:, 1:]
            X_clean = add_constant(X_clean)
            
            # Run OLS
            model = OLS(y_clean, X_clean)
            fit = model.fit()
            
            # Newey-West HAC standard errors
            cov_hac_matrix = cov_hac(fit, nlags=NEWEY_WEST_LAGS)
            se_hac = np.sqrt(np.diag(cov_hac_matrix))
            
            # Extract shock coefficient
            coef = fit.params[shock_var]
            stderr = se_hac[1]  # Index 1 after constant
            tstat = coef / stderr
            pval = 2 * (1 - stats.t.cdf(np.abs(tstat), df=len(data) - len(X_cols) - 1))
            
            ci_lower = coef - 1.96 * stderr
            ci_upper = coef + 1.96 * stderr
            
            # Store results
            results['horizons'].append(h)
            results['coef'].append(coef)
            results['stderr'].append(stderr)
            results['tstat'].append(tstat)
            results['pval'].append(pval)
            results['ci_lower'].append(ci_lower)
            results['ci_upper'].append(ci_upper)
            results['n_obs'].append(len(data))
            
        except Exception as e:
            # Handle any errors
            results['horizons'].append(h)
            results['coef'].append(np.nan)
            results['stderr'].append(np.nan)
            results['tstat'].append(np.nan)
            results['pval'].append(np.nan)
            results['ci_lower'].append(np.nan)
            results['ci_upper'].append(np.nan)
            results['n_obs'].append(0)
    
    return results

# ==============================================================================
# 7. Run LP for All Sectors
# ==============================================================================
print("🚀 Running Local Projections for all sectors...")
print()

all_results = []

# Process OECD sectors
print("Processing OECD sectors...")
for i, sector in enumerate(oecd_sectors, 1):
    print(f"   [{i}/{len(oecd_sectors)}] {sector}...", end='\r')
    result = run_local_projections_sector(df, sector, HORIZONS, shock_var, control_vars)
    if result:
        result['source'] = 'OECD'
        all_results.append(result)

print(f"\n   ✓ Completed {len([r for r in all_results if r['source']=='OECD'])} OECD sectors")
print()

# Process ADB sectors
print("Processing ADB sectors...")
for i, sector in enumerate(adb_sectors, 1):
    print(f"   [{i}/{len(adb_sectors)}] {sector}...", end='\r')
    result = run_local_projections_sector(df, sector, HORIZONS, shock_var, control_vars)
    if result:
        result['source'] = 'ADB'
        all_results.append(result)

print(f"\n   ✓ Completed {len([r for r in all_results if r['source']=='ADB'])} ADB sectors")
print()

print(f"Total sectors analyzed: {len(all_results)}")
print()

# ==============================================================================
# 8. Create Coefficient and P-value Matrices
# ==============================================================================
print("📋 Creating result matrices...")

# Coefficient matrix
coef_data = []
pval_data = []
stderr_data = []

for result in all_results:
    coef_row = {'sector': result['sector'], 'source': result['source']}
    pval_row = {'sector': result['sector'], 'source': result['source']}
    stderr_row = {'sector': result['sector'], 'source': result['source']}
    
    for i, h in enumerate(result['horizons']):
        coef_row[f'h{h}'] = result['coef'][i]
        pval_row[f'h{h}'] = result['pval'][i]
        stderr_row[f'h{h}'] = result['stderr'][i]
    
    coef_data.append(coef_row)
    pval_data.append(pval_row)
    stderr_data.append(stderr_row)

coef_matrix = pd.DataFrame(coef_data)
pval_matrix = pd.DataFrame(pval_data)
stderr_matrix = pd.DataFrame(stderr_data)

# Save matrices
coef_file = OUTPUT_DIR / 'sectoral_lp_coefficients.csv'
pval_file = OUTPUT_DIR / 'sectoral_lp_pvalues.csv'
stderr_file = OUTPUT_DIR / 'sectoral_lp_stderr.csv'

coef_matrix.to_csv(coef_file, index=False)
pval_matrix.to_csv(pval_file, index=False)
stderr_matrix.to_csv(stderr_file, index=False)

print(f"   ✓ Coefficient matrix saved: {coef_file}")
print(f"   ✓ P-value matrix saved: {pval_file}")
print(f"   ✓ Std error matrix saved: {stderr_file}")
print()

# ==============================================================================
# 9. Identify Significant Sectors at h=0
# ==============================================================================
print("🎯 Identifying significant sectors at h=0...")

sig_sectors = []

for result in all_results:
    if len(result['horizons']) > 0 and result['horizons'][0] == 0:
        h0_idx = 0
        coef = result['coef'][h0_idx]
        pval = result['pval'][h0_idx]
        stderr = result['stderr'][h0_idx]
        
        if not np.isnan(pval):
            sig_level = ''
            if pval < 0.01:
                sig_level = '***'
            elif pval < 0.05:
                sig_level = '**'
            elif pval < 0.10:
                sig_level = '*'
            
            if sig_level:  # Significant
                sig_sectors.append({
                    'sector': result['sector'],
                    'source': result['source'],
                    'coef': coef,
                    'stderr': stderr,
                    'pval': pval,
                    'sig': sig_level,
                    'direction': 'Positive' if coef > 0 else 'Negative'
                })

sig_df = pd.DataFrame(sig_sectors)
sig_df = sig_df.sort_values('pval')

sig_file = OUTPUT_DIR / 'sectoral_significant_h0.csv'
sig_df.to_csv(sig_file, index=False)

print(f"   ✓ Found {len(sig_sectors)} significant sectors at h=0")
print(f"   ✓ Saved to: {sig_file}")
print()

# Display top 10
if len(sig_df) > 0:
    print("Top 10 Most Significant Sectors at h=0:")
    print("-" * 80)
    for idx, row in sig_df.head(10).iterrows():
        print(f"   {row['sector']:15s} [{row['source']}] β={row['coef']:7.4f} {row['sig']} "
              f"(p={row['pval']:.4f}) {row['direction']}")
    print()

# ==============================================================================
# 10. Create Heatmap - Top 20 Significant Sectors
# ==============================================================================
print("📈 Creating impact magnitude heatmap...")

# Get top 20 by absolute coefficient at h=0
top_sectors = sig_df.nlargest(20, 'coef')['sector'].tolist() if len(sig_df) >= 20 else sig_df['sector'].tolist()

if len(top_sectors) > 0:
    # Filter coefficient matrix for top sectors
    heatmap_data = coef_matrix[coef_matrix['sector'].isin(top_sectors)].copy()
    heatmap_data = heatmap_data.set_index('sector')
    heatmap_data = heatmap_data.drop(columns=['source'], errors='ignore')
    
    # Create figure
    plt.figure(figsize=(10, max(8, len(top_sectors) * 0.4)))
    
    # Create heatmap
    sns.heatmap(
        heatmap_data,
        cmap='RdBu_r',
        center=0,
        annot=True,
        fmt='.3f',
        cbar_kws={'label': 'Coefficient'},
        linewidths=0.5,
        linecolor='gray'
    )
    
    plt.title('Impact Magnitude: Top 20 Significant Sectors\n(US Monetary Policy Shock Response)',
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Horizon (years)', fontsize=12, fontweight='bold')
    plt.ylabel('Sector', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save
    heatmap_file = OUTPUT_DIR / 'sectoral_impact_heatmap_top20.png'
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Heatmap saved: {heatmap_file}")
    print()
else:
    print("   ⚠ No significant sectors found for heatmap")
    print()

# ==============================================================================
# 11. Summary Statistics
# ==============================================================================
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

# Overall statistics
total_oecd = len([r for r in all_results if r['source'] == 'OECD'])
total_adb = len([r for r in all_results if r['source'] == 'ADB'])

sig_oecd = len(sig_df[sig_df['source'] == 'OECD']) if len(sig_df) > 0 else 0
sig_adb = len(sig_df[sig_df['source'] == 'ADB']) if len(sig_df) > 0 else 0

print(f"Total sectors analyzed:")
print(f"   OECD: {total_oecd}")
print(f"   ADB:  {total_adb}")
print(f"   Total: {total_oecd + total_adb}")
print()

print(f"Significant sectors at h=0:")
print(f"   OECD: {sig_oecd} ({sig_oecd/total_oecd*100:.1f}% of OECD)" if total_oecd > 0 else "   OECD: 0")
print(f"   ADB:  {sig_adb} ({sig_adb/total_adb*100:.1f}% of ADB)" if total_adb > 0 else "   ADB: 0")
print(f"   Total: {len(sig_df)}")
print()

# Direction analysis
if len(sig_df) > 0:
    positive = len(sig_df[sig_df['direction'] == 'Positive'])
    negative = len(sig_df[sig_df['direction'] == 'Negative'])
    
    print(f"Impact direction (significant sectors):")
    print(f"   Positive: {positive} ({positive/len(sig_df)*100:.1f}%)")
    print(f"   Negative: {negative} ({negative/len(sig_df)*100:.1f}%)")
    print()

# Significance levels
if len(sig_df) > 0:
    sig_001 = len(sig_df[sig_df['sig'] == '***'])
    sig_005 = len(sig_df[sig_df['sig'] == '**'])
    sig_010 = len(sig_df[sig_df['sig'] == '*'])
    
    print(f"Significance levels:")
    print(f"   1% (***):  {sig_001}")
    print(f"   5% (**):   {sig_005}")
    print(f"   10% (*):   {sig_010}")
    print()

print("=" * 80)
print("✓ Sectoral analysis completed successfully!")
print("=" * 80)
print()

print("Output files:")
print(f"   1. {coef_file.name}")
print(f"   2. {pval_file.name}")
print(f"   3. {stderr_file.name}")
print(f"   4. {sig_file.name}")
if len(top_sectors) > 0:
    print(f"   5. {heatmap_file.name}")
