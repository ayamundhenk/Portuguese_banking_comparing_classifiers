# Portuguese Banking — Classifier Comparison

## Overview
This repository compares classifiers on the UCI Portuguese banking dataset to predict whether a customer subscribes to a term deposit (`y`: yes/no).

## Problem Statement
A Portuguese bank conducts telemarketing campaigns to sell term deposit products. The goal is to predict whether a customer will subscribe to a term deposit (y = yes/no) and determine which machine learning classifier performs best.

## How I ran the analysis
- Script: `run_analysis.py` — downloads the dataset, preprocesses, runs CV evaluation for baseline models, fits the best model, and saves artifacts.
- To reproduce locally:

```bash
python3 -m pip install --user pandas scikit-learn joblib matplotlib seaborn
python3 run_analysis.py
```

## Results (baseline)
Cross-validation mean metrics (5-fold):

- LogisticRegression: accuracy=0.9097, precision=0.6571, recall=0.4151, f1=0.5088, roc_auc=0.9331
- RandomForest:       accuracy=0.9106, precision=0.6444, recall=0.4615, f1=0.5375, roc_auc=0.9411
- GradientBoosting:   accuracy=0.9154, precision=0.6569, recall=0.5218, f1=0.5814, roc_auc=0.9448

Best model by CV F1: `GradientBoosting`.

Test set (held-out) performance for `GradientBoosting`:

- Accuracy: 0.9219
- Precision (class=1): 0.6971
- Recall (class=1): 0.5431
- F1 (class=1): 0.6105
- ROC AUC: 0.9532

## Artifacts
- `results_cv.csv` — CV summary table
- `best_model.joblib` — fitted best model
- `scaler.joblib` — fitted scaler for numeric features
- `data/` — downloaded dataset files

## Next steps
- Hyperparameter tuning (Grid/Randomized search)
- More feature engineering and target encoding for categorical variables
- Handle class imbalance (class weights, SMOTE) and evaluate business-cost-sensitive metrics


