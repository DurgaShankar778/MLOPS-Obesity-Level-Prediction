# Obesity Level Prediction - MLOps Project

Multiclass classification of obesity level (7 classes) from eating habits and physical
condition. The project follows the same Lab 1-5 MLOps structure as the Telco Churn project
and reuses its ideas (reproducible split, saved preprocessors, MLflow tracking, schema
validation, sklearn `ColumnTransformer` pipeline), adapted to the obesity dataset.

## Structure

```
MLOPS-PROJECT/
├── data/
│   ├── raw/ObesityDataSet_raw_and_data_sinthetic.csv   # Lab 1 - raw dataset (2,111 rows, 17 columns)
│   └── processed/                                      # Lab 1 / Lab 5 outputs (generated)
├── notebooks/project_implementation.ipynb              # Lab 1 - EDA, model comparison, tuning
├── src/
│   ├── common.py                    # shared paths, class order, cleaning + split helpers
│   ├── preprocess.py                # Lab 1 - clean, BMI, split, scale/encode
│   ├── train.py                     # Lab 2 - baseline Random Forest
│   ├── evaluate.py                  # Lab 2 - metrics + error analysis
│   ├── train_mlflow.py              # Lab 4 - MLflow tracking, 5 runs
│   ├── validate_reproducibility.py  # Lab 4
│   ├── validate_data.py             # Lab 5 - pandera schema check
│   ├── preprocess_pipeline.py       # Lab 5 - ColumnTransformer pipeline
│   ├── validate_outputs.py          # Lab 5 - output checks + report
│   └── predict.py                   # predict the level of a new person
├── pipelines/
│   ├── run_lab3_baseline.py
│   ├── run_lab4_tracking.py
│   └── run_lab5_pipeline.py
├── paper/Obesity_Level_Prediction_Full_Paper.docx
├── models/      # generated: scaler.pkl, ohe.pkl, preprocessor.pkl, *.pkl model files
├── outputs/     # generated: misclassified_samples.csv, per_class_metrics.csv
├── artifacts/   # generated: confusion matrices, ROC curves, JSON reports
├── logs/
├── mlruns/      # Lab 4 - MLflow tracking store (generated)
├── requirements.txt
└── .gitignore
```

## What was adapted from the churn project

| Churn project | This project |
|---|---|
| Binary target `Churn` (Yes/No) | 7-class target `NObeyesdad`, encoded 0-6 from lightest to heaviest |
| Drops `customerID`, fixes `TotalCharges` | Removes 24 duplicate rows (2,111 -> 2,087), adds `BMI = Weight / Height²` |
| Precision / recall / F1 | Weighted precision / recall / F1, macro F1, one-vs-rest ROC-AUC |
| False negatives / false positives CSVs | `misclassified_samples.csv` (with actual vs predicted level) and `per_class_metrics.csv` |
| Single ROC curve | One-vs-rest ROC curve per class, confusion matrix with class names |
| One MLflow run | Five runs: Random Forest, tuned Random Forest (GridSearchCV), Decision Tree, Logistic Regression, KNN |
| Telco pandera schema | Obesity schema (ranges for Age, Height, Weight, FCVC, NCP, CH2O, FAF, TUE; allowed values for every categorical column) |
| Impute `TotalCharges` with 0 | Median for numeric, most-frequent for categorical (as in the notebook) |

## Getting started

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Lab 1-3: preprocessing -> baseline training -> evaluation
python pipelines/run_lab3_baseline.py

# Lab 4: preprocessing -> MLflow tracking (5 runs) -> reproducibility check
python pipelines/run_lab4_tracking.py
mlflow ui                     # compare the runs in the browser

# Lab 5: schema validation -> sklearn pipeline preprocessing -> output checks
python pipelines/run_lab5_pipeline.py

# Predict the obesity level of a new person (edit the example inside the script)
python src/predict.py
```

Run every command from the project root. The Lab 5 pipeline overwrites the same
`data/processed/*.npy` arrays as Lab 1 (the features are identical) and additionally saves
`models/preprocessor.pkl`.

## Results (reference run: Python 3.12, scikit-learn 1.8)

Stratified 80/20 split (`random_state=42`): 1,669 train rows, 418 test rows, 24 features after encoding.

| Model | Accuracy | Weighted F1 |
|---|---|---|
| Random Forest (200 trees) | 0.9856 | 0.9856 |
| Random Forest (tuned) | 0.9856 | 0.9856 |
| Decision Tree | 0.9689 | 0.9688 |
| Logistic Regression | 0.9067 | 0.9062 |
| KNN (k=5) | 0.8780 | 0.8693 |

Your numbers should be the same or very close; small differences can come from library versions.

## Notes

- **BMI and the target are closely related.** The obesity level in this dataset is defined from
  BMI, so `Weight`, `Height` and the engineered `BMI` make the task easy for tree models. Treat
  the high accuracy as a property of the dataset, not as evidence of a hard problem solved.
- **77% of the rows are synthetic** (SMOTE), which is why the classes are almost perfectly balanced.
- Duplicates are removed *before* the split so identical rows cannot appear in both train and test.

## Git workflow (Lab 3)

```bash
git init
git add . && git commit -m "Initial obesity MLOps project"
git checkout -b feature/mlflow-tracking
# ... work, commit ...
git checkout main && git merge feature/mlflow-tracking
git tag -a v1.0 -m "Labs 1-5 complete"
```
