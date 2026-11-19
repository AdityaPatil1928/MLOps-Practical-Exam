# Automated Machine Learning Pipeline Architecture

## 1. Overview
The automated pipeline orchestrates end-to-end data processing, feature engineering, statistical schema validation, and model evaluation using DVC (`dvc.yaml`). Every stage tracks its dependencies and outputs, enabling intelligent caching and automated invalidation when code or upstream data changes.

---

## 2. Directed Acyclic Graph (DAG)

```
        +----------------------------------------+
        |                collect                 |
        |  (deps: collect.py, dataset_loader.py) |
        +-------------------+--------------------+
                            |
                            | (data/raw/raw_data.csv)
                            v
        +----------------------------------------+
        |               preprocess               |
        |  (deps: preprocess.py, raw_data.csv)   |
        +-------------------+--------------------+
                            |
                            | (data/processed/preprocessed_data.csv)
                            v
        +----------------------------------------+
        |                features                |
        |  (deps: features.py, preprocessed.csv) |
        +-------------------+--------------------+
                            |
                            | (data/processed/featured_data.csv)
                            v
           +----------------+----------------+
           |                                 |
           v                                 v
+----------------------+         +-------------------------------------+
|       validate       |         |             train_eval              |
| (data integrity check|         | (RandomForestClassifier, metrics,   |
|  sys.exit(1) barrier)|         |  pipeline_model.joblib)             |
+----------------------+         +-------------------------------------+
```

---

## 3. Pipeline Stages Specification

| Stage Name | Purpose & Logic | Inputs / Dependencies | Outputs / Artifacts | Validation & Integrity Rules |
| :--- | :--- | :--- | :--- | :--- |
| **`collect`** | Ingest active dataset from `dataset_loader.py` and attach UTC ISO collection timestamp. | `src/pipeline/collect.py`, `dataset_loader.py` | `data/raw/raw_data.csv` | Dataset must load successfully without runtime errors. |
| **`preprocess`** | Deduplicate rows, coerce features to numeric, impute nulls with median, drop invalid targets. | `src/pipeline/preprocess.py`, `data/raw/raw_data.csv` | `data/processed/preprocessed_data.csv` | Deduplication ensures uniqueness; `collected_at` column stripped. |
| **`features`** | Dynamically detect non-constant numeric features and engineer 3 domain features. | `src/pipeline/features.py`, `data/processed/preprocessed_data.csv` | `data/processed/featured_data.csv` | Generates interaction product, ratio (+epsilon), and quantile bins. |
| **`validate`** | Strict quality gate: checks schema, minimum row/column counts, zero nulls, zero infs, target class validity. | `src/pipeline/validate.py`, `data/processed/featured_data.csv` | *Quality Gate (No file output)* | Raises `DataValidationError` and calls `sys.exit(1)` on any violation. |
| **`train_eval`** | Stratified 80/20 train/test split, trains `RandomForestClassifier(100)`, computes macro metrics. | `src/pipeline/train_eval.py`, `data/processed/featured_data.csv` | `models/pipeline_model.joblib`, `metrics/eval_metrics.json` | Computes `accuracy`, `precision_macro`, `recall_macro`, and `f1_macro`. |

---

## 4. DVC Caching & Reproducibility Mechanics
- When `dvc repro` is triggered, DVC hashes each stage's dependencies and command string.
- If hashes match `dvc.lock`, the stage execution is skipped: `Stage '<stage>' didn't change, skipping`.
- If any input or code file is altered, DVC runs only the invalidated stage and downstream dependents.
