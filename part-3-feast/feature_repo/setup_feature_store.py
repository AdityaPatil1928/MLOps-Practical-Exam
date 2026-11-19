import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path to import dataset_loader
REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR.parent))
from dataset_loader import get_dataset, DATASET_NAME

def setup_feature_repo():
    print(f"=== [Feast Setup] Setting up Feature Store for dataset: '{DATASET_NAME}' ===")
    df, raw_feature_cols, target_col = get_dataset()
    
    # Identify first two non-constant numeric columns for feature engineering
    candidate_cols = [c for c in raw_feature_cols if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1]
    if len(candidate_cols) >= 2:
        col_a, col_b = candidate_cols[0], candidate_cols[1]
    else:
        col_a, col_b = raw_feature_cols[0], raw_feature_cols[0]
        
    print(f"Base columns selected for interaction: '{col_a}', '{col_b}'")
    
    # Compute 2 engineered features
    df["feat_interaction_prod"] = (df[col_a] * df[col_b]).astype(np.float64)
    df["feat_interaction_ratio"] = (df[col_a] / (df[col_b].abs() + 1e-5)).astype(np.float64)
    engineered_cols = ["feat_interaction_prod", "feat_interaction_ratio"]
    
    # Cast all feature columns to float64 for Feast / Parquet consistency
    for col in raw_feature_cols:
        df[col] = df[col].astype(np.float64)
        
    # Add Entity ID
    df["sample_id"] = np.arange(len(df), dtype=np.int64)
    
    # Add UTC event_timestamp and created_timestamp
    base_time = datetime.now(timezone.utc) - timedelta(hours=2)
    timestamps = [base_time + timedelta(seconds=i * 10) for i in range(len(df))]
    df["event_timestamp"] = pd.to_datetime(timestamps)
    df["created_timestamp"] = pd.to_datetime(timestamps)
    
    # Ensure data directory exists
    data_dir = REPO_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = data_dir / "features.parquet"
    
    # Write to Parquet
    df.to_parquet(parquet_path, index=False)
    print(f"Wrote {len(df)} rows to Parquet source: {parquet_path}")
    
    # Generate features.py dynamically
    features_py_content = f'''# Dynamically generated Feast Feature Definitions
from datetime import timedelta
from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.value_type import ValueType
from feast.types import Float64

# 1. Define Entity
sample_entity = Entity(
    name="sample_id",
    join_keys=["sample_id"],
    value_type=ValueType.INT64,
    description="Unique identifier for each sample instance",
)

# 2. Define Batch Source
features_source = FileSource(
    name="features_source",
    path="data/features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# 3. Define Raw Features FeatureView
raw_features_fv = FeatureView(
    name="raw_features_fv",
    entities=[sample_entity],
    ttl=timedelta(days=365),
    schema=[
'''
    for col in raw_feature_cols:
        features_py_content += f'        Field(name="{col}", dtype=Float64),\n'
        
    features_py_content += f'''    ],
    online=True,
    source=features_source,
    tags={{"tier": "raw", "dataset": "{DATASET_NAME}"}},
)

# 4. Define Engineered Features FeatureView
engineered_features_fv = FeatureView(
    name="engineered_features_fv",
    entities=[sample_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="feat_interaction_prod", dtype=Float64),
        Field(name="feat_interaction_ratio", dtype=Float64),
    ],
    online=True,
    source=features_source,
    tags={{"tier": "engineered"}},
)

# 5. Define FeatureService for Cross-Model Reusability
exam_feature_service = FeatureService(
    name="exam_feature_service",
    features=[raw_features_fv, engineered_features_fv],
    tags={{"team": "mlops-exam"}},
)
'''
    features_py_path = REPO_DIR / "features.py"
    with open(features_py_path, "w", encoding="utf-8") as f:
        f.write(features_py_content)
        
    print(f"Successfully generated Feast definitions at: {features_py_path}")
    print(f" - Entity: sample_id")
    print(f" - FeatureView 1: raw_features_fv ({len(raw_feature_cols)} fields)")
    print(f" - FeatureView 2: engineered_features_fv (2 fields)")
    print(f" - FeatureService: exam_feature_service")
    print(f"\nNext step: run 'feast apply' from inside feature_repo/ directory.")

if __name__ == "__main__":
    setup_feature_repo()
