# Part 4: MLflow Experiment Tracking & Model Registry

## 1. Overview & Learning Objectives
This module provides hands-on mastery of **MLflow**:
- Configuring an MLflow Tracking Server with a persistent SQLite database backend.
- Training and logging multiple candidate model architectures with parameters, metrics, confusion matrix figures, and serialized model artifacts.
- Evaluating and comparing experiment runs visually and programmatically.
- Selecting the best-performing model based on test macro F1-score and registering it in the Model Registry.
- Promoting the registered model to the `Staging` lifecycle stage and loading it for inference.

---

## 2. Git Submission Workflow (For Group Exam Repo)
> **NOTE:** Do NOT run `git init` inside this directory during local practice. Git repository management occurs in your team's cloned exam repo.

1. Clone group exam repo: `git clone https://github.com/<owner>/MLOps-Practical-Exam.git` and `cd MLOps-Practical-Exam`
2. Sync `main`: `git switch main` then `git pull origin main`
3. Create feature branch: `git switch -c part-4-mlflow`
4. Copy the entire `part-4-mlflow/` folder into the cloned repository.
5. Execute the verification steps below, stage changes (`git add .`), commit (`git commit -m "feat(mlflow): tracking server and model registry"`), push (`git push -u origin part-4-mlflow`), and create a Pull Request into `main`.

---

## 3. Step-by-Step Execution Guide (`cmd.exe`)

This module uses a 2-terminal workflow in Windows Command Prompt (`cmd.exe`):
- **Terminal 1:** Runs the MLflow Tracking Server.
- **Terminal 2:** Executes the experiment training, registration, and inference scripts.

---

### Terminal 1: Start MLflow Tracking Server
Open `cmd.exe`, navigate to `part-4-mlflow`, activate `.venv-mlflow`, and launch the server:
```cmd
cd "c:\Users\PRATIK\Desktop\Aaditya Patil\MLops_Practical\part-4-mlflow"
call ..\envs\.venv-mlflow\Scripts\activate.bat
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```
> Keep Terminal 1 open and running! You will see request logs as scripts interact with the server.

---

### Terminal 2: Train, Register & Run Inference
Open a **second** `cmd.exe` window:
```cmd
cd "c:\Users\PRATIK\Desktop\Aaditya Patil\MLops_Practical\part-4-mlflow"
call ..\envs\.venv-mlflow\Scripts\activate.bat
set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

#### Step 3.1: Configure Active Dataset (Optional)
Check `dataset_loader.py` to confirm `DATASET_NAME` matches the examiner's dataset (`"wine"`, `"breast_cancer"`, or `"digits"`):
```cmd
python dataset_loader.py
```

#### Step 3.2: Train Candidate Models & Track with MLflow
Train Logistic Regression, Random Forest (Shallow), and Random Forest (Deep):
```cmd
python src/train_with_mlflow.py
```
This logs:
- Hyperparameters (`n_estimators`, `max_depth`, `C`, `scaler`)
- Metrics (`accuracy`, `precision_macro`, `recall_macro`, `f1_macro`)
- Artifacts: Confusion matrix plots (`visualizations/confusion_matrix_<model>.png`)
- Serialized Scikit-Learn models with input signatures

#### Step 3.3: Inspect Runs in the MLflow UI
Open your browser and navigate to:
```
http://127.0.0.1:5000
```
- Click on experiment **`exam-classification-baseline`**.
- Compare the three runs side-by-side using the metric charts.
- Click on any individual run to view its parameters, metrics, and confusion matrix image.

#### Step 3.4: Register Best Model & Promote to Staging
Query the tracking server for the highest macro F1-score run, register it under `exam-classifier-prod`, and transition it to `Staging`:
```cmd
python src/register_best_model.py
```
In the MLflow UI, click the **Models** tab at the top to inspect `exam-classifier-prod` with version 1 in `Staging`.

#### Step 3.5: Load Registered Model from Staging & Run Inference
Load the staging model directly via its model URI (`models:/exam-classifier-prod/Staging`) and test inference:
```cmd
python src/load_registered_model.py
```
**Expected Output:** Prints the loaded estimator class and a formatted table of 5 test rows with actual vs. predicted labels.

---

## 4. Reset to Initial Clean State

> **IMPORTANT:** In **Terminal 1**, press `Ctrl+C` to stop the MLflow server before running the cleanup commands below. If the server is running, Windows will lock `mlflow.db`.

In Terminal 2 (or after stopping the server):
```cmd
if exist "mlflow.db" del /q /f "mlflow.db"
if exist "confusion_matrix_*.png" del /q /f "confusion_matrix_*.png"
if exist "mlruns" rmdir /s /q "mlruns"
if exist "mlartifacts" rmdir /s /q "mlartifacts"
if exist "src\__pycache__" rmdir /s /q "src\__pycache__"
```
