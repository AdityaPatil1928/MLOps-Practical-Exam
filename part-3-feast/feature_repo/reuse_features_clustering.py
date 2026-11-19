import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from feast import FeatureStore

def cluster_with_reused_features():
    repo_path = Path(__file__).resolve().parent
    print(f"=== [Feast Feature Reuse] Cross-Model Feature Reuse with KMeans Clustering ===")
    store = FeatureStore(repo_path=str(repo_path))
    
    parquet_path = repo_path / "data" / "features.parquet"
    if not parquet_path.exists():
        print(f"[ERROR] '{parquet_path}' does not exist.")
        sys.exit(1)
        
    source_df = pd.read_parquet(parquet_path)
    num_samples = len(source_df)
    
    # Query feature service for all entities
    entity_df = pd.DataFrame({
        "sample_id": np.arange(num_samples, dtype=np.int64),
        "timestamp": [datetime.now(timezone.utc)] * num_samples,
    })
    
    print(f"Retrieving shared features from 'exam_feature_service' for {num_samples} entities...")
    feature_service = store.get_feature_service("exam_feature_service")
    training_data = store.get_historical_features(
        entity_df=entity_df,
        features=feature_service,
    )
    df_features = training_data.to_df()
    
    # Drop entity and timestamp columns to get pure feature matrix
    drop_cols = ["sample_id", "timestamp"]
    feature_matrix = df_features.drop(columns=[c for c in drop_cols if c in df_features.columns])
    
    # Fill any NaNs if present
    feature_matrix.fillna(0.0, inplace=True)
    
    print(f"Feature matrix ready: {feature_matrix.shape} (reused across models without recomputing features)")
    print(f"Includes engineered features: 'feat_interaction_prod', 'feat_interaction_ratio'")
    
    # Train KMeans clustering model (3 clusters)
    k = 3
    print(f"\nTraining KMeans (k={k}, random_state=42)...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(feature_matrix)
    
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    print("\n--- Clustering Results (Cluster Distribution) ---")
    for cluster_id, count in cluster_counts.items():
        print(f"  Cluster {cluster_id}: {count} samples ({count/len(clusters)*100:.1f}%)")
        
    print("\n[SUCCESS] Demonstrated feature reuse: Feature definitions maintained centrally in Feast were reused seamlessly in an unsupervised clustering model!")

if __name__ == "__main__":
    cluster_with_reused_features()
