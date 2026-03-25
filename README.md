# Diabetes Prediction using Machine Learning

A complete, reproducible machine learning pipeline to predict diabetes presence in the Pima Indians population using the Pima Indians Diabetes Database.

## Project Overview

This project implements and compares multiple supervised learning models for binary classification of diabetes (presence vs absence). The workflow includes:

- Data loading and exploratory data analysis (EDA)
- Feature scaling and class imbalance handling with SMOTE
- Model training with hyperparameter tuning (GridSearchCV)
- Cross-validation and test set evaluation
- Ensemble methods (Voting and Stacking classifiers)
- Figure generation and logging

The goal is to build a clinically meaningful model that balances sensitivity (recall) and precision, suitable as a decision support tool.

## Dataset Description

- **Name:** Pima Indians Diabetes Database  
- **Samples:** 768 female patients (age ≥ 21)  
- **Features (8):**
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
- **Target:** `Outcome` (0 = Non-diabetic, 1 = Diabetic)
- **Class distribution:**
  - No Diabetes: 500 (65.1%)
  - Diabetes: 268 (34.9%)
  - Imbalance ratio: 1.87 : 1

## Methodology

1. **Data Preprocessing**
   - Stratified train-test split (75% / 25%)
   - Standardization with `StandardScaler` (fit on train, applied to test)
   - SMOTE applied to training data only to balance classes (1:1)

2. **Models**
   - Logistic Regression (baseline, interpretable)
   - Random Forest
   - Gradient Boosting
   - XGBoost
   - SVM with RBF kernel

3. **Hyperparameter Tuning**
   - `GridSearchCV` with 5-fold stratified cross-validation
   - Scoring metric: ROC-AUC

4. **Evaluation Metrics**
   - Accuracy
   - Precision
   - Recall (Sensitivity)
   - F1-Score
   - ROC-AUC
   - Confusion matrix and classification report

5. **Ensemble Methods**
   - Soft Voting Classifier (combining tuned models)
   - Stacking Classifier with Logistic Regression as meta-learner

## Results (Summary)

On the held-out test set (192 samples):

- **Logistic Regression**
  - Accuracy: ~0.80
  - Precision: ~0.69
  - Recall: ~0.75
  - ROC-AUC: ~0.87

- **Random Forest**
  - Highest recall (~0.78), slightly lower precision

- **All models**
  - Accuracy > 0.75
  - ROC-AUC > 0.82

Logistic Regression achieved the best overall generalization on the test set, despite some ensemble models having higher cross-validation scores.

## Repository Structure

```text
diabetes-ml-prediction/
├─ src/
│  ├─ config.py
│  ├─ logging_utils.py
│  ├─ data_loader.py
│  ├─ preprocessing.py
│  ├─ feature_engineering.py
│  ├─ modeling.py
│  ├─ evaluation.py
│  ├─ visualization.py
├─ notebooks/
│  ├─ 01_exploratory_data_analysis.ipynb
│  ├─ 02_model_training_and_evaluation.ipynb
│  ├─ 03_ensembles_and_discussion.ipynb
├─ data/
│  ├─ raw/
│  ├─ processed/
├─ reports/
│  ├─ figures/
├─ main.py
├─ generate_processed.py
├─ requirements.txt
├─ .gitignore
├─ README.md
```

## How to Run the Project

### 1. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Place the raw dataset

```text
data/raw/diabetes.csv
```

(Download from Kaggle: Pima Indians Diabetes Database.)

### 4. Generate processed data

```bash
python generate_processed.py
```

### 5. Run the full pipeline

```bash
python main.py
```

This will:

- Load raw data
- Generate EDA figures in `reports/figures/`
- Preprocess and balance data
- Tune models and run cross-validation
- Evaluate on the test set
- Train and evaluate ensemble models
- Log all steps to `project.log`

### 6. Reproduce Results

To reproduce the reported results:

1. Use the same random seed (`251000861`) as defined in `src/config.py`.
2. Ensure the same train-test split and SMOTE configuration.
3. Run `generate_processed.py` followed by `main.py`.
4. Compare metrics printed in the console and logged in `project.log`.

## Notebooks

- `01_exploratory_data_analysis.ipynb`  
  Mirrors the Introduction and Dataset/EDA sections.

- `02_model_training_and_evaluation.ipynb`  
  Covers preprocessing, model training, hyperparameter tuning, and evaluation.

- `03_ensembles_and_discussion.ipynb`  
  Implements ensemble methods and summarizes discussion and conclusions.

Use these notebooks for interactive exploration and figure regeneration.
