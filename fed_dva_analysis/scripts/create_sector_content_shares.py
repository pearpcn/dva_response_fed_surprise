"""
Create sector content share table (Thailand, 2022) from OECD DOMIMP file.

Outputs:
- fed_dva_analysis/outputs/sector_content_shares_2022.csv

Definitions (per sector i):
- domestic_content_share = VA_i / X_i
- import_content_share = M_i / X_i
- export_content_share = EXPO_i / X_i
- domestic_intermediate_share = DOM_input_i / X_i
- total_intermediate_share = (DOM_input_i + M_i) / X_i

Where:
- X_i comes from OUTPUT row
- VA_i comes from VALU row
- M_i is sum of IMP_* rows into sector column i
- DOM_input_i is sum of DOM_* rows into sector column i
- EXPO_i is EXPO final-demand column for DOM_i row
"""

from pathlib import Path
import numpy as np
import pandas as pd


def main() -> None:
    workspace = Path(__file__).resolve().parents[2]
    data_file = workspace / "data_raw" / "IOTs_DOMIMP" / "THA2022dom.csv"
    output_file = workspace / "fed_dva_analysis" / "outputs" / "sector_content_shares_2022.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not data_file.exists():
        raise FileNotFoundError(f"Missing input file: {data_file}")

    df = pd.read_csv(data_file, index_col=0)

    fd_keywords = {
        "HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "DPABR", "CONS_NONRES", "EXPO", "TOTAL"
    }
    sector_cols = [col for col in df.columns if col not in fd_keywords and col != ".."]

    dom_rows = [idx for idx in df.index if str(idx).startswith("DOM_")]
    imp_rows = [idx for idx in df.index if str(idx).startswith("IMP_")]

    if "VALU" not in df.index or "OUTPUT" not in df.index:
        raise ValueError("Expected VALU and OUTPUT rows in THA2022dom.csv")

    if "EXPO" not in df.columns:
        raise ValueError("Expected EXPO column in THA2022dom.csv")

    x = df.loc["OUTPUT", sector_cols].astype(float)
    va = df.loc["VALU", sector_cols].astype(float)

    dom_inputs = df.loc[dom_rows, sector_cols].astype(float).sum(axis=0)
    imports = (
        df.loc[imp_rows, sector_cols].astype(float).sum(axis=0)
        if len(imp_rows) > 0
        else pd.Series(0.0, index=sector_cols)
    )

    exports_series = df.loc[dom_rows, "EXPO"].astype(float)
    export_by_sector = pd.Series(index=sector_cols, dtype=float)
    for row_name, value in exports_series.items():
        sector = str(row_name).replace("DOM_", "")
        if sector in export_by_sector.index:
            export_by_sector.loc[sector] = value
    export_by_sector = export_by_sector.fillna(0.0)

    denom = x.replace(0, np.nan)

    out = pd.DataFrame(
        {
            "sector": sector_cols,
            "output": x.values,
            "domestic_content_share": (va / denom).values,
            "import_content_share": (imports / denom).values,
            "export_content_share": (export_by_sector / denom).values,
            "domestic_intermediate_share": (dom_inputs / denom).values,
            "total_intermediate_share": ((dom_inputs + imports) / denom).values,
        }
    )

    out["content_plus_total_intermediate"] = (
        out["domestic_content_share"] + out["total_intermediate_share"]
    )

    out.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")
    print(f"Rows: {len(out)}")
    print(
        "Mean shares | "
        f"Domestic: {out['domestic_content_share'].mean():.4f}, "
        f"Import: {out['import_content_share'].mean():.4f}, "
        f"Export: {out['export_content_share'].mean():.4f}, "
        f"Total intermediate: {out['total_intermediate_share'].mean():.4f}"
    )


if __name__ == "__main__":
    main()
