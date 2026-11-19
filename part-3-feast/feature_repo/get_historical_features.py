import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np
from feast import FeatureStore

def fetch_historical_features():
    repo_path = Path(__file__).resolve().parent
    print(f"=== [Feast Offline Retrieval] Point-in-Time Correct Historical Features ===")
    store = FeatureStore(repo_path=str(repo_path))
    
    # Read Parquet to see available sample_ids
    parquet_path = repo_path / "data" / "features.parquet"
    if not parquet_path.exists():
        print(f"[ERROR] '{parquet_path}' does not exist. Run setup_feature_store.py and feast apply first.")
        sys.exit(1)
        
    source_df = pd.read_parquet(parquet_path)
    num_samples = min(25, len(source_df))
    
    # Construct entity DataFrame with query timestamps (point-in-time)
    entity_df = pd.DataFrame({
        "sample_id": np.arange(num_samples, dtype=np.int64),
        "timestamp": [datetime.now(timezone.utc)] * num_samples,
    })
    
    print(f"Querying historical features for {num_samples} entities via 'exam_feature_service'...")
    feature_service = store.get_feature_service("exam_feature_service")
    
    training_data = store.get_historical_features(
        entity_df=entity_df,
        features=feature_service,
    )
    
    training_df = training_data.to_df()
    print("\n--- Point-in-Time Feature Retrieval Complete ---")
    print(f"Training DataFrame Shape: {training_df.shape}")
    print(f"Columns ({len(training_df.columns)}): {list(training_df.columns[:8])}... (truncated)")
    print("\nPreview (first 3 rows):")
    print(training_df.head(3).to_string())
    return training_df

if __name__ == "__main__":
    fetch_historical_features()
