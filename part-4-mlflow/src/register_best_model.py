import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

def register_best():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    print(f"=== [Model Registry] Connecting to MLflow: {tracking_uri} ===")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    
    experiment_name = "exam-classification-baseline"
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"[ERROR] Experiment '{experiment_name}' not found. Please run train_with_mlflow.py first.")
        sys.exit(1)
        
    print(f"Searching runs for experiment '{experiment_name}' (ID: {experiment.experiment_id})...")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_macro DESC"],
        max_results=5,
    )
    
    if not runs:
        print(f"[ERROR] No runs found in experiment '{experiment_name}'.")
        sys.exit(1)
        
    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_f1 = best_run.data.metrics.get("f1_macro", 0.0)
    best_acc = best_run.data.metrics.get("accuracy", 0.0)
    run_name = best_run.data.tags.get("mlflow.runName", "unknown")
    
    print("\n--- Best Run Identification ---")
    print(f"  Best Run Name: {run_name}")
    print(f"  Run ID:        {best_run_id}")
    print(f"  F1 (Macro):    {best_f1}")
    print(f"  Accuracy:      {best_acc}")
    
    model_name = "exam-classifier-prod"
    model_uri = f"runs:/{best_run_id}/model"
    print(f"\nRegistering model from '{model_uri}' to Model Registry as '{model_name}'...")
    
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"Successfully registered model '{model_name}' Version {mv.version}.")
    
    # Transition Model Version to "Staging"
    print(f"Promoting Version {mv.version} to Stage 'Staging'...")
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Staging",
        archive_existing_versions=False,
    )
    
    # Add description / metadata
    client.update_model_version(
        name=model_name,
        version=mv.version,
        description=f"Promoted {run_name} to Staging based on highest test F1-macro ({best_f1})."
    )
    
    print(f"\n[SUCCESS] Model '{model_name}' Version {mv.version} is now in 'Staging'!")
    print(f"Callable production URI: 'models:/{model_name}/Staging'")

if __name__ == "__main__":
    register_best()
