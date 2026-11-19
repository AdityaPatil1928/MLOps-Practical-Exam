import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def generate_features(input_path: str, output_path: str):
    print(f"=== [Stage 3: Feature Engineering] Engineering features for {input_path} ===")
    in_file = Path(input_path)
    if not in_file.exists():
        print(f"[ERROR] Input file {input_path} does not exist.")
        sys.exit(1)
        
    df = pd.read_csv(in_file)
    target_col = "target_label"
    
    # Identify non-constant numeric feature columns
    candidate_cols = []
    for col in df.columns:
        if col != target_col and pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() > 1:
                candidate_cols.append(col)
                
    if len(candidate_cols) < 2:
        print("[WARNING] Fewer than 2 non-constant numeric columns found. Using any available numeric columns.")
        candidate_cols = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
        
    if len(candidate_cols) >= 2:
        col_a, col_b = candidate_cols[0], candidate_cols[1]
    elif len(candidate_cols) == 1:
        col_a = candidate_cols[0]
        col_b = candidate_cols[0]
    else:
        print("[ERROR] No numeric feature columns available for feature engineering.")
        sys.exit(1)
        
    print(f"Selected base features for interaction: '{col_a}' and '{col_b}'")
    
    # 1. Interaction Product Feature
    df["feat_interaction_prod"] = df[col_a] * df[col_b]
    
    # 2. Interaction Ratio Feature (with epsilon to avoid division by zero)
    df["feat_interaction_ratio"] = df[col_a] / (df[col_b].abs() + 1e-5)
    
    # 3. Quantile Binning Feature
    try:
        df["feat_quantile_bin"] = pd.qcut(df[col_a], q=4, labels=False, duplicates="drop").astype(float)
    except Exception:
        # Fallback to cut if qcut has issues with duplicate bins
        df["feat_quantile_bin"] = pd.cut(df[col_a], bins=4, labels=False).fillna(0).astype(float)
        
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    
    print("Features engineered successfully:")
    print(" - feat_interaction_prod")
    print(" - feat_interaction_ratio")
    print(" - feat_quantile_bin")
    print(f"Total columns now: {len(df.columns)}")
    print(f"Output saved to: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Feature Engineering")
    parser.add_argument("--input", type=str, default="data/processed/preprocessed_data.csv", help="Input preprocessed CSV path")
    parser.add_argument("--output", type=str, default="data/processed/featured_data.csv", help="Output featured CSV path")
    args = parser.parse_args()
    generate_features(args.input, args.output)
