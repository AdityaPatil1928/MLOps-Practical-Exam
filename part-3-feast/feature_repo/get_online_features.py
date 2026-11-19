import os
import sys
from pathlib import Path
import pandas as pd
from feast import FeatureStore

def fetch_online_features():
    repo_path = Path(__file__).resolve().parent
    print(f"=== [Feast Online Retrieval] Querying SQLite Online Store ===")
    store = FeatureStore(repo_path=str(repo_path))
    
    entity_rows = [
        {"sample_id": 0},
        {"sample_id": 1},
        {"sample_id": 2},
    ]
    
    print(f"Requesting online features for sample_ids: [0, 1, 2] via 'exam_feature_service'...")
    feature_service = store.get_feature_service("exam_feature_service")
    
    response = store.get_online_features(
        features=feature_service,
        entity_rows=entity_rows,
    )
    
    df_online = pd.DataFrame(response.to_dict())
    print("\n--- Online Features Retrieved (Low Latency) ---")
    print(f"Retrieved DataFrame Shape: {df_online.shape}")
    print(df_online.to_string(index=False))
    return df_online

if __name__ == "__main__":
    fetch_online_features()
