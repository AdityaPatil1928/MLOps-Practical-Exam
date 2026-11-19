import os
import sys
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset_loader import get_dataset, DATASET_NAME

def load_and_infer():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    print(f"=== [Model Inference] Connecting to MLflow: {tracking_uri} ===")
    mlflow.set_tracking_uri(tracking_uri)
    
    model_uri = "models:/exam-classifier-prod/Staging"
    print(f"Loading registered model from URI: '{model_uri}'...")
    
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Model successfully loaded into memory!")
        print(f"Loaded Estimator Type: {type(model)}")
    except Exception as e:
        print(f"[ERROR] Failed to load model from '{model_uri}'. Error: {e}")
        print("Ensure the MLflow tracking server is running and register_best_model.py was executed.")
        sys.exit(1)
        
    # Load 5 sample rows from active dataset
    print(f"\nLoading sample data from active dataset: '{DATASET_NAME}'...")
    df, feature_cols, target_col = get_dataset()
    sample_df = df.iloc[:5]
    X_samples = sample_df[feature_cols]
    actual_labels = sample_df[target_col].tolist()
    
    # Run Predictions
    predictions = model.predict(X_samples)
    
    # Format and Display Results
    results_df = pd.DataFrame({
        "Sample": [f"Row_{i}" for i in range(len(sample_df))],
        "Actual_Label": actual_labels,
        "Predicted_Label": predictions,
        "Match": [a == p for a, p in zip(actual_labels, predictions)],
    })
    
    print("\n" + "=" * 55)
    print(f"INFERENCE RESULTS FROM 'models:/exam-classifier-prod/Staging'")
    print("=" * 55)
    print(results_df.to_string(index=False))
    print("=" * 55)

if __name__ == "__main__":
    load_and_infer()
