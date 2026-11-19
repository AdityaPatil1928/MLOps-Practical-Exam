import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

class DataValidationError(Exception):
    """Custom exception raised when dataset fails quality, schema, or integrity validation."""
    pass

def validate(input_path: str):
    print(f"=== [Stage 4: Data Validation] Validating {input_path} ===")
    in_file = Path(input_path)
    if not in_file.exists():
        raise DataValidationError(f"Target data file does not exist: {input_path}")
        
    df = pd.read_csv(in_file)
    target_col = "target_label"
    
    # 1. Validate Target Column Existence
    if target_col not in df.columns:
        raise DataValidationError(f"Missing mandatory target column: '{target_col}'")
        
    # 2. Validate Minimum Row and Feature Counts
    if len(df) < 10:
        raise DataValidationError(f"Dataset has too few rows ({len(df)} rows). Expected >= 10.")
        
    feature_cols = [c for c in df.columns if c != target_col]
    if len(feature_cols) < 2:
        raise DataValidationError(f"Dataset has insufficient features ({len(feature_cols)}). Expected >= 2.")
        
    # 3. Validate Zero Missing / Null Values
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        raise DataValidationError(f"Unexpected null values detected in columns: {cols_with_nulls.to_dict()}")
        
    # 4. Validate Target Label Integrity
    unique_targets = df[target_col].dropna().unique()
    if len(unique_targets) < 2:
        raise DataValidationError(f"Target column must contain at least 2 distinct classes. Found: {unique_targets}")
        
    # Check for empty or blank string targets
    if any(str(t).strip() == "" for t in unique_targets):
        raise DataValidationError("Target column contains empty or whitespace-only labels.")
        
    # 5. Validate Numeric Integrity (No Infs or NaN)
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            if np.isinf(df[col]).any():
                raise DataValidationError(f"Infinite values (+inf or -inf) detected in numeric column '{col}'")
            # Check for corrupted extreme floating values (e.g. > 1e12 or < -1e12)
            if (df[col].abs() > 1e12).any():
                raise DataValidationError(f"Extreme corrupted numerical values (> 1e12) detected in column '{col}'")
                
    # 6. Validate Engineered Feature Presence
    expected_engineered = ["feat_interaction_prod", "feat_interaction_ratio", "feat_quantile_bin"]
    for feat in expected_engineered:
        if feat not in df.columns:
            raise DataValidationError(f"Required engineered feature '{feat}' is missing from dataset.")
            
    print("[PASS] Data Validation Successful!")
    print(f" - Verified rows: {len(df)}")
    print(f" - Verified columns: {len(df.columns)}")
    print(f" - Verified classes: {list(unique_targets)}")
    print(" - Zero nulls, zero infs, and all schema constraints satisfied.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: Data Validation")
    parser.add_argument("--input", type=str, default="data/processed/featured_data.csv", help="Input featured CSV path")
    args = parser.parse_args()
    try:
        validate(args.input)
    except DataValidationError as e:
        print(f"\n[FATAL VALIDATION ERROR] {e}", file=sys.stderr)
        print("[HARD STOP] Halting ML pipeline execution due to data integrity violation.", file=sys.stderr)
        sys.exit(1)
