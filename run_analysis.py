import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)
zip_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip'
zip_path = os.path.join(DATA_DIR, 'bank-additional.zip')
csv_path = os.path.join(DATA_DIR, 'bank-additional', 'bank-additional-full.csv')

if not os.path.exists(csv_path):
    print('Downloading dataset...')
    urllib.request.urlretrieve(zip_url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(DATA_DIR)
    print('Downloaded and extracted to', DATA_DIR)
else:
    print('Dataset already present at', csv_path)

# Load
df = pd.read_csv(csv_path, sep=';')
print('Rows, columns:', df.shape)

# Preprocess
df2 = df.copy()
df2['target'] = (df2['y'] == 'yes').astype(int)
X = df2.drop(columns=['y', 'target'])
y = df2['target']
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])
print('Training shape:', X_train.shape)

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42)
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
results = []
for name, model in models.items():
    print('Evaluating', name)
    cv_res = cross_validate(model, X_train, y_train, cv=skf, scoring=scoring, return_train_score=False)
    summary = {metric: np.mean(cv_res['test_' + metric]) for metric in scoring}
    summary['model'] = name
    results.append(summary)
res_df = pd.DataFrame(results).set_index('model')
print('\nCross-validation results:')
print(res_df)
res_df.to_csv('results_cv.csv')

# Fit best model and evaluate
best_name = res_df['f1'].idxmax()
best_model = models[best_name]
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
print('\nBest model:', best_name)
print(classification_report(y_test, y_pred, digits=4))
if y_proba is not None:
    print('Test ROC AUC:', roc_auc_score(y_test, y_proba))

# Save model and scaler
joblib.dump(best_model, 'best_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print('Saved results to results_cv.csv, best_model.joblib, scaler.joblib')
