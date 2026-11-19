# Part 3: Feature Store Implementation (Feast)

## 1. Overview & Learning Objectives
This module provides hands-on mastery of **Feast (Feature Store)**:
- Establishing a Feast feature repository (`feature_store.yaml`).
- Defining Entities, FeatureViews, and FeatureServices.
- Generating offline Parquet storage and materializing features into a SQLite online store.
- Retrieving low-latency online features and point-in-time-correct historical training sets.
- Demonstrating cross-model feature reuse across supervised and unsupervised workflows.

---

## 2. Git Submission Workflow (For Group Exam Repo)
> **NOTE:** Do NOT run `git init` inside this directory during local practice. Git repository management occurs in your team's cloned exam repo.

1. Clone group exam repo: `git clone https://github.com/<owner>/MLOps-Practical-Exam.git` and `cd MLOps-Practical-Exam`
2. Sync `main`: `git switch main` then `git pull origin main`
3. Create feature branch: `git switch -c part-3-feast`
4. Copy the entire `part-3-feast/` folder into the cloned repository.
5. Execute the verification steps below, stage changes (`git add .`), commit (`git commit -m "feat(feast): feature store repository and materialization"`), push (`git push -u origin part-3-feast`), and create a Pull Request into `main`.

---

## 3. Step-by-Step Execution Guide (`cmd.exe`)

Open Windows Command Prompt (`cmd.exe`) and navigate into `part-3-feast`:
```cmd
cd "c:\Users\PRATIK\Desktop\Aaditya Patil\MLops_Practical\part-3-feast"
```

### Step 3.1: Activate Virtual Environment
Activate the `.venv-feast` environment:
```cmd
call ..\envs\.venv-feast\Scripts\activate.bat
```

### Step 3.2: Configure Runtime Dataset (Optional)
Open `dataset_loader.py` and set `DATASET_NAME` to `"wine"`, `"breast_cancer"`, or `"digits"`:
```python
DATASET_NAME = "wine"
```
Verify the dataset loader:
```cmd
python dataset_loader.py
```

### Step 3.3: Set Up Feature Store & Generate Parquet Data
Navigate into `feature_repo` and generate the Parquet offline source and dynamic `features.py`:
```cmd
cd feature_repo
python setup_feature_store.py
```
This produces:
- `feature_repo/data/features.parquet`
- `feature_repo/features.py` (Entities, FeatureViews, FeatureService)

### Step 3.4: Register Definitions with Feast (`feast apply`)
Apply the feature definitions to the registry:
```cmd
feast apply
```
Verify registered components:
```cmd
feast entities list
feast feature-views list
feast feature-services list
```

### Step 3.5: Materialize Features into SQLite Online Store
Run the materialization script to populate the SQLite online store:
```cmd
python materialize.py
```
This writes the latest feature state to `feature_repo/data/online_store.db`.

### Step 3.6: Low-Latency Online Feature Retrieval
Retrieve real-time features for sample instances (simulating live production scoring):
```cmd
python get_online_features.py
```

### Step 3.7: Point-in-Time Correct Historical Retrieval
Retrieve offline training features with point-in-time correctness (preventing data leakage):
```cmd
python get_historical_features.py
```

### Step 3.8: Demonstrate Cross-Model Feature Reuse
Demonstrate how another model (unsupervised KMeans clustering) consumes the exact same centralized feature definitions without rewriting feature engineering code:
```cmd
python reuse_features_clustering.py
```

Return to `part-3-feast` root when finished:
```cmd
cd ..
```

---

## 4. Reset to Initial Clean State

To wipe all generated Parquet files, SQLite databases, and dynamically generated feature definitions before a new test run or dataset switch:
```cmd
if exist "feature_repo\data" rmdir /s /q "feature_repo\data"
if exist "feature_repo\features.py" del /q /f "feature_repo\features.py"
if exist "feature_repo\__pycache__" rmdir /s /q "feature_repo\__pycache__"
```
