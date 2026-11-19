import argparse
import json
import os
import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def train_eval(input_path: str, model_out: str, metrics_out: str):
    print(f"=== [Stage 5: Train & Evaluate] Training model from {input_path} ===")
    in_file = Path(input_path)
    if not in_file.exists():
        print(f"[ERROR] Input file {input_path} does not exist.")
        sys.exit(1)
        
    df = pd.read_csv(in_file)
    target_col = "target_label"
    feature_cols = [c for c in df.columns if c != target_col]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Dataset split: Train={X_train.shape[0]} samples, Test={X_test.shape[0]} samples")
    
    # Train Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = clf.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "num_features": int(len(feature_cols))
    }
    
    # Save Model
    m_out = Path(model_out)
    m_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, m_out)
    print(f"Model saved to: {m_out}")
    
    # Save Metrics JSON
    met_out = Path(metrics_out)
    met_out.parent.mkdir(parents=True, exist_ok=True)
    with open(met_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"Evaluation metrics written to: {met_out}")
    
    print("\n--- Model Evaluation Summary ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 5: Model Training and Evaluation")
    parser.add_argument("--input", type=str, default="data/processed/featured_data.csv", help="Input featured CSV path")
    parser.add_argument("--model-out", type=str, default="models/pipeline_model.joblib", help="Output model path")
    parser.add_argument("--metrics-out", type=str, default="metrics/eval_metrics.json", help="Output metrics path")
    args = parser.parse_args()
    train_eval(args.input, args.model_out, args.metrics_out)
