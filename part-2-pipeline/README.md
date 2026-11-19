# Part 2: Automated Machine Learning Pipeline

## 1. Overview & Learning Objectives
This module implements an automated, reproducible, modular ML pipeline managed via **DVC (`dvc.yaml`)**:
- 5 distinct stages: `collect` -> `preprocess` -> `features` -> `validate` -> `train_eval`.
- Statistical and schema validation with hard-stop failure handling (`sys.exit(1)`).
- Dependency tracking, DAG visualization (`dvc dag`), and evaluation metrics tracking (`dvc metrics show`).
- Pipeline caching verification ensuring unchanged stages are skipped.

---

## 2. Git Submission Workflow (For Group Exam Repo)
> **NOTE:** Do NOT run `git init` inside this directory during local practice. Git repository management occurs in your team's cloned exam repo.

1. Clone group exam repo: `git clone https://github.com/<owner>/MLOps-Practical-Exam.git` and `cd MLOps-Practical-Exam`
2. Sync `main`: `git switch main` then `git pull origin main`
3. Create feature branch: `git switch -c part-2-pipeline`
4. Copy the entire `part-2-pipeline/` folder into the cloned repository.
5. Execute the verification steps below, stage changes (`git add .`), commit (`git commit -m "feat(pipeline): automated 5-stage DVC ML pipeline"`), push (`git push -u origin part-2-pipeline`), and merge via PR into `main`.

---

## 3. Step-by-Step Execution Guide (`cmd.exe`)

Open Windows Command Prompt (`cmd.exe`) and navigate into `part-2-pipeline`:
```cmd
cd "c:\Users\PRATIK\Desktop\Aaditya Patil\MLops_Practical\part-2-pipeline"
```

### Step 3.1: Activate Virtual Environment
Activate the `.venv-base` environment:
```cmd
call ..\envs\.venv-base\Scripts\activate.bat
```

### Step 3.2: Configure Runtime Dataset (Optional)
Open `dataset_loader.py` and set `DATASET_NAME` to the examiner's assigned dataset (`"wine"`, `"breast_cancer"`, or `"digits"`):
```python
DATASET_NAME = "wine"
```
Verify dataset loading:
```cmd
python dataset_loader.py
```

### Step 3.3: Initialize DVC
If inside a cloned Git repository:
```cmd
dvc init
```
*(If testing standalone outside of a Git repo:)*
```cmd
dvc init --no-scm
```

### Step 3.4: Inspect the Pipeline DAG
Visualize the stage dependencies:
```cmd
dvc dag
```

### Step 3.5: Execute the Full Pipeline (`dvc repro`)
Run all 5 stages from ingestion to training:
```cmd
dvc repro
```
This produces:
- `data/raw/raw_data.csv`
- `data/processed/preprocessed_data.csv`
- `data/processed/featured_data.csv`
- `models/pipeline_model.joblib`
- `metrics/eval_metrics.json`
- `dvc.lock`

### Step 3.6: Verify Caching Mechanics
Execute `dvc repro` a second time immediately:
```cmd
dvc repro
```
**Expected Output:**
```
Data and pipelines are up to date.
```
DVC detects that no code or input data changed, executing 0 stages and guaranteeing efficiency.

### Step 3.7: Inspect Pipeline Metrics
```cmd
dvc metrics show
```
Example output:
```
metrics\eval_metrics.json:
        accuracy: 1.0
        f1_macro: 1.0
        num_features: 16
        precision_macro: 1.0
        recall_macro: 1.0
        test_samples: 36
        train_samples: 142
```

### Step 3.8: Test Validation Hard-Stop (`sys.exit(1)`)
To test that bad data triggers an immediate failure and aborts training:
```cmd
:: Corrupt the featured dataset by injecting an invalid null
python -c "import pandas as pd; df=pd.read_csv('data/processed/featured_data.csv'); df.loc[0, 'target_label']=None; df.to_csv('data/processed/featured_data.csv', index=False)"

:: Run validation directly to observe hard stop
python src/pipeline/validate.py --input data/processed/featured_data.csv
```
**Expected Output:** `[FATAL VALIDATION ERROR] Unexpected null values... [HARD STOP]` with exit code `1`.

Now restore clean data by running:
```cmd
python src/pipeline/features.py
python src/pipeline/validate.py
```

---

## 4. Reset to Initial Clean State

To wipe all generated data, models, metrics, and DVC state before testing or switching datasets:
```cmd
if exist "data" rmdir /s /q "data"
if exist "models" rmdir /s /q "models"
if exist "metrics" rmdir /s /q "metrics"
if exist ".dvc" rmdir /s /q ".dvc"
if exist ".dvcignore" del /q /f ".dvcignore"
if exist "dvc.lock" del /q /f "dvc.lock"
```
