import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def preprocess(input_path: str, output_path: str):
    print(f"=== [Stage 2: Preprocess] Processing {input_path} ===")
    in_file = Path(input_path)
    if not in_file.exists():
        print(f"[ERROR] Input file {input_path} does not exist.")
        sys.exit(1)
        
    df = pd.read_csv(in_file)
    initial_rows = len(df)
    target_col = "target_label"
    
    # 1. Drop duplicates cleanly
    df = df.drop_duplicates()
    dedup_rows = len(df)
    
    # 2. Drop rows with missing target_label
    if target_col in df.columns:
        df = df.dropna(subset=[target_col])
        
    # 3. Drop collection timestamp column if present
    if "collected_at" in df.columns:
        df = df.drop(columns=["collected_at"])
        
    # 4. Identify feature columns and coerce to numeric
    feature_cols = [c for c in df.columns if c != target_col]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Impute missing values with column median
        median_val = df[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        df[col] = df[col].fillna(median_val)
        
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    
    print(f"Preprocessing completed.")
    print(f"Initial rows: {initial_rows} -> Clean rows: {len(df)} (Deduplicated: {initial_rows - dedup_rows})")
    print(f"Output saved to: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Data Preprocessing")
    parser.add_argument("--input", type=str, default="data/raw/raw_data.csv", help="Input raw CSV path")
    parser.add_argument("--output", type=str, default="data/processed/preprocessed_data.csv", help="Output preprocessed CSV path")
    args = parser.parse_args()
    preprocess(args.input, args.output)
