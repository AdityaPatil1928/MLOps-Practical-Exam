import os
import sys
from pathlib import Path

# Ensure matplotlib uses a local writable directory for font cache
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset_loader import get_dataset, DATASET_NAME

def train_and_track():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    print(f"=== [MLflow Tracking] Connecting to Tracking URI: {tracking_uri} ===")
    mlflow.set_tracking_uri(tracking_uri)
    
    experiment_name = "exam-classification-baseline"
    mlflow.set_experiment(experiment_name)
    print(f"Experiment: '{experiment_name}'")
    
    # Load Active Dataset
    print(f"Loading active dataset: '{DATASET_NAME}'...")
    df, feature_cols, target_col = get_dataset()
    X = df[feature_cols]
    y = df[target_col]
    
    # 80/20 Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data Split: Train={len(X_train)} samples, Test={len(X_test)} samples, Features={len(feature_cols)}")
    
    # Define 3 Candidate Model Configurations
    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        ),
        "random_forest_shallow": RandomForestClassifier(
            n_estimators=50, max_depth=3, random_state=42
        ),
        "random_forest_deep": RandomForestClassifier(
            n_estimators=100, max_depth=None, random_state=42
        ),
    }
    
    model_params = {
        "logistic_regression": {
            "model_type": "LogisticRegression",
            "scaler": "StandardScaler",
            "C": 1.0,
            "max_iter": 1000,
            "dataset": DATASET_NAME,
        },
        "random_forest_shallow": {
            "model_type": "RandomForestClassifier",
            "n_estimators": 50,
            "max_depth": 3,
            "dataset": DATASET_NAME,
        },
        "random_forest_deep": {
            "model_type": "RandomForestClassifier",
            "n_estimators": 100,
            "max_depth": "None",
            "dataset": DATASET_NAME,
        },
    }
    
    run_summaries = []
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            print(f"\n--- Training Candidate Model: {name} (Run ID: {run.info.run_id}) ---")
            
            # Log Parameters
            mlflow.log_params(model_params[name])
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            
            # Fit Model
            model.fit(X_train, y_train)
            
            # Evaluate Model
            y_pred = model.predict(X_test)
            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
            rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
            f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
            
            metrics = {
                "accuracy": round(acc, 4),
                "precision_macro": round(prec, 4),
                "recall_macro": round(rec, 4),
                "f1_macro": round(f1, 4),
            }
            mlflow.log_metrics(metrics)
            print(f"Metrics logged: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
            
            # Generate & Log Confusion Matrix Plot Artifact
            cm_fig_path = f"confusion_matrix_{name}.png"
            fig, ax = plt.subplots(figsize=(6, 5))
            cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
            disp.plot(ax=ax, cmap="Blues", values_format="d")
            plt.title(f"Confusion Matrix - {name} ({DATASET_NAME})")
            plt.tight_layout()
            plt.savefig(cm_fig_path, dpi=120)
            plt.close(fig)
            
            mlflow.log_artifact(cm_fig_path, artifact_path="visualizations")
            print(f"Logged artifact: {cm_fig_path}")
            
            # Clean up local plot file after logging
            if os.path.exists(cm_fig_path):
                os.remove(cm_fig_path)
                
            # Log Serialized Model Artifact with Input Example
            input_example = X_train.iloc[:2]
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=input_example,
            )
            print("Logged serialized model artifact with signature & input example.")
            
            run_summaries.append({
                "model_name": name,
                "run_id": run.info.run_id,
                "accuracy": round(acc, 4),
                "f1_macro": round(f1, 4),
            })
            
    # Print Comparison Table
    summary_df = pd.DataFrame(run_summaries)
    print("\n" + "=" * 65)
    print(f"EXPERIMENT RUN COMPARISON: '{experiment_name}'")
    print("=" * 65)
    print(summary_df.to_string(index=False))
    print("=" * 65)

if __name__ == "__main__":
    train_and_track()
