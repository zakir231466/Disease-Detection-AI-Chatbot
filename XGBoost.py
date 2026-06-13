import re
import random
import pandas as pd
import numpy as np
import csv
from pathlib import Path
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from difflib import get_close_matches
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")

# ------------------ Load Data ------------------
def resolve_base_dir():
    script_dir = Path(__file__).resolve().parent
    candidates = [script_dir, script_dir.parent]

    for c in candidates:
        if (c / "Data").exists() and (c / "Master Data").exists():
            return c

    raise FileNotFoundError("Data folders not found")

BASE_DIR = resolve_base_dir()

training = pd.read_csv(BASE_DIR / "Data" / "Training.csv")
testing = pd.read_csv(BASE_DIR / "Data" / "Testing.csv")

training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
testing.columns = testing.columns.str.replace(r"\.\d+$", "", regex=True)

training = training.loc[:, ~training.columns.duplicated()]
testing = testing.loc[:, ~testing.columns.duplicated()]
training = training.drop_duplicates()

# ------------------ Features ------------------
cols = training.columns[:-1]
x = training[cols]
y = training["prognosis"]

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# ------------------ XGBOOST ------------------
print("🚀 Training XGBoost Model...")

model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(np.unique(y)),
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=1,
    reg_lambda=2,
    eval_metric="mlogloss",
    n_jobs=-1,
    random_state=42
)

model.fit(x_train, y_train)

# ------------------ Evaluation ------------------
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

train_acc = accuracy_score(y_train, y_train_pred) * 100
test_acc = accuracy_score(y_test, y_test_pred) * 100

precision = precision_score(y_test, y_test_pred, average="weighted", zero_division=0) * 100
recall = recall_score(y_test, y_test_pred, average="weighted", zero_division=0) * 100
f1 = f1_score(y_test, y_test_pred, average="weighted", zero_division=0) * 100

cv_scores = cross_val_score(model, x, y, cv=5)

print("\n======================================")
print("🚀 XGBoost Model")
print(f"📊 Training Accuracy : {train_acc:.2f}%")
print(f"📊 Testing Accuracy  : {test_acc:.2f}%")
print(f"🔁 CV Accuracy       : {cv_scores.mean()*100:.2f}%")
print(f"🎯 Precision         : {precision:.2f}%")
print(f"📢 Recall            : {recall:.2f}%")
print(f"🏆 F1-score          : {f1:.2f}%")
print("======================================\n")

# ------------------ DATA DICTS ------------------
severityDictionary = {}
description_list = {}
precautionDictionary = {}
symptoms_dict = {s: i for i, s in enumerate(cols)}

def getDescription():
    with open(BASE_DIR / "Master Data" / "symptom_Description.csv") as f:
        for r in csv.reader(f):
            if len(r) >= 2:
                description_list[r[0]] = r[1]

def getSeverityDict():
    with open(BASE_DIR / "Master Data" / "Symptom_severity.csv") as f:
        for r in csv.reader(f):
            if len(r) >= 2:
                try:
                    severityDictionary[r[0]] = int(r[1])
                except:
                    pass

def getprecautionDict():
    with open(BASE_DIR / "Master Data" / "symptom_precaution.csv") as f:
        for r in csv.reader(f):
            if len(r) >= 5:
                precautionDictionary[r[0]] = [r[1], r[2], r[3], r[4]]

# ------------------ SYMPTOM MATCH ------------------
symptom_synonyms = {
    "stomach ache": "stomach_pain",
    "belly pain": "stomach_pain",
    "loose motion": "diarrhea",
    "high fever": "fever",
    "coughing": "cough",
    "breathing issue": "breathlessness",
    "body pain": "muscle_pain",
}

def extract_symptoms(text, all_symptoms):
    found = []
    text = text.lower()

    for k, v in symptom_synonyms.items():
        if k in text:
            found.append(v)

    for s in all_symptoms:
        if s.replace("_", " ") in text:
            found.append(s)

    words = re.findall(r"\w+", text)

    for w in words:
        close = get_close_matches(
            w,
            [s.replace("_", " ") for s in all_symptoms],
            n=1,
            cutoff=0.8
        )
        if close:
            for s in all_symptoms:
                if s.replace("_", " ") == close[0]:
                    found.append(s)

    return list(set(found))

# ------------------ PREDICTION ------------------
def predict(symptoms):
    vec = np.zeros(len(symptoms_dict))

    for s in symptoms:
        if s in symptoms_dict:
            vec[symptoms_dict[s]] = 1

    df = pd.DataFrame([vec], columns=cols)

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0]

    disease = le.inverse_transform([pred])[0]
    confidence = round(prob[pred] * 100, 2)

    return disease, confidence

# ------------------ CHATBOT ------------------
def chatbot():
    getDescription()
    getSeverityDict()
    getprecautionDict()

    print("🤖 HealthCare ChatBot\n")

    name = input("👉 What is your name? ")
    age = input("👉 Please enter your age: ")
    gender = input("👉 What is your gender? (M/F/Other): ")

    symptoms_input = input("👉 Describe your symptoms: ")

    symptoms = extract_symptoms(symptoms_input, cols)

    if not symptoms:
        print("❌ No symptoms detected")
        return

    print("✅ Symptoms detected:", symptoms)

    days = input("👉 How many days? : ")
    severity = input("👉 Severity (1-10): ")
    pre = input("👉 Pre-existing conditions? : ")
    lifestyle = input("👉 Smoking/drinking/sleep issues? : ")
    family = input("👉 Family history? : ")

    disease, confidence = predict(symptoms)

    print("\n----------- RESULT -----------")
    print("🩺 Disease:", disease)
    print("🔎 Confidence:", confidence, "%")

    print("\n📖 Description:")
    print(description_list.get(disease, "No description available"))

    if disease in precautionDictionary:
        print("\n🛡️ Precautions:")
        for i, p in enumerate(precautionDictionary[disease], 1):
            print(i, p)

    print("\n💡 Stay healthy,", name)

# ------------------ RUN ------------------
if __name__ == "__main__":
    chatbot()