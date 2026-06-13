import pandas as pd
import numpy as np
import warnings
from pathlib import Path

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# =========================
# BASE PATH
# =========================
BASE_DIR = Path(__file__).resolve().parent

# =========================
# LOAD DATA
# =========================
training = pd.read_csv(BASE_DIR / "Data" / "Training.csv")

training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
training = training.loc[:, ~training.columns.duplicated()]

# =========================
# FEATURES / LABEL
# =========================
X = training.iloc[:, :-1]
y = training["prognosis"]

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

# =========================
# SPLIT DATA
# =========================
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# =========================
# MODEL
# =========================
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)

print("🚀 Training XGBoost Model...")
model.fit(x_train, y_train)

# =========================
# PREDICTION
# =========================
y_pred = model.predict(x_test)

# =========================
# METRICS
# =========================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

cv_scores = cross_val_score(model, X, y, cv=5)

# =========================
# CLASSIFICATION REPORT (CLEAN)
# =========================
target_names = le.inverse_transform(np.arange(len(le.classes_)))

report = classification_report(
    y_test,
    y_pred,
    target_names=target_names,
    digits=2,
    zero_division=0
)

# =========================
# PRINT REPORT
# =========================
print("\n==============================")
print("XGBOOST EVALUATION REPORT")
print("==============================")

print(f"Accuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"F1-Score  : {f1*100:.2f}%")

print("\n5-FOLD CROSS VALIDATION")
print("------------------------------")
print(f"Mean Accuracy : {cv_scores.mean()*100:.2f}%")
print(f"Std Dev       : {cv_scores.std()*100:.2f}%")

print("\nClassification Evaluation Metrics Report:")
print("=" * 65)
print(report)
print("=" * 65)

# =========================
# SAVE REPORT
# =========================
report_text = f"""
XGBOOST EVALUATION REPORT
==============================

Accuracy  : {accuracy*100:.2f}%
Precision : {precision*100:.2f}%
Recall    : {recall*100:.2f}%
F1-Score  : {f1*100:.2f}%

5-FOLD CROSS VALIDATION
------------------------------
Mean Accuracy : {cv_scores.mean()*100:.2f}%
Std Dev       : {cv_scores.std()*100:.2f}%

CLASSIFICATION REPORT
------------------------------
{report}
"""

with open("xgboost_evaluation_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

print("\n✅ Saved: xgboost_evaluation_report.txt")
print("==============================")