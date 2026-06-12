import re
import random
import pandas as pd
import numpy as np
import csv
from pathlib import Path
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from difflib import get_close_matches
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------ Load Data ------------------
def resolve_base_dir():
    script_dir = Path(__file__).resolve().parent
    candidates = [script_dir, script_dir.parent]

    for candidate in candidates:
        if (candidate / "Data").exists() and (candidate / "Master Data").exists():
            return candidate

    raise FileNotFoundError("Could not find 'Data' and 'Master Data' folders.")

BASE_DIR = resolve_base_dir()

training = pd.read_csv(BASE_DIR / "Data" / "Training.csv")
testing = pd.read_csv(BASE_DIR / "Data" / "Testing.csv")

# ------------------ Clean duplicate columns ------------------
training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
testing.columns = testing.columns.str.replace(r"\.\d+$", "", regex=True)

training = training.loc[:, ~training.columns.duplicated()]
testing = testing.loc[:, ~testing.columns.duplicated()]

# ------------------ Features & labels ------------------
cols = training.columns[:-1]
x = training[cols]
y = training["prognosis"]

# ------------------ Encode labels ------------------
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

# ------------------ Train/test split ------------------
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ------------------ Train XGBoost model ------------------
print("🚀 Training XGBoost model...")

num_classes = len(np.unique(y))

model = XGBClassifier(
    objective="multi:softprob",
    num_class=num_classes,
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
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

# ------------------ Dictionaries ------------------
severityDictionary = {}
description_list = {}
precautionDictionary = {}

symptoms_dict = {symptom: idx for idx, symptom in enumerate(cols)}

def getDescription():
    with open(BASE_DIR / "Master Data" / "symptom_Description.csv") as csv_file:
        for row in csv.reader(csv_file):
            if len(row) >= 2:
                description_list[row[0]] = row[1]

def getSeverityDict():
    with open(BASE_DIR / "Master Data" / "Symptom_severity.csv") as csv_file:
        for row in csv.reader(csv_file):
            if len(row) < 2:
                continue
            try:
                severityDictionary[row[0]] = int(row[1])
            except:
                pass

def getprecautionDict():
    with open(BASE_DIR / "Master Data" / "symptom_precaution.csv") as csv_file:
        for row in csv.reader(csv_file):
            if len(row) >= 5:
                precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

# ------------------ Symptom synonyms ------------------
symptom_synonyms = {
    "stomach ache": "stomach_pain",
    "belly pain": "stomach_pain",
    "tummy pain": "stomach_pain",
    "loose motion": "diarrhea",
    "motions": "diarrhea",
    "high temperature": "fever",
    "temperature": "fever",
    "feaver": "fever",
    "coughing": "cough",
    "throat pain": "sore_throat",
    "cold": "chills",
    "breathing issue": "breathlessness",
    "shortness of breath": "breathlessness",
    "body ache": "muscle_pain",
}

# ------------------ Extract symptoms ------------------
def extract_symptoms(user_input, all_symptoms):
    extracted = []
    text = user_input.lower().replace("-", " ")

    for phrase, mapped in symptom_synonyms.items():
        if phrase in text:
            extracted.append(mapped)

    for symptom in all_symptoms:
        if symptom.replace("_", " ") in text:
            extracted.append(symptom)

    words = re.findall(r"\w+", text)

    for word in words:
        close = get_close_matches(
            word,
            [s.replace("_", " ") for s in all_symptoms],
            n=1,
            cutoff=0.8
        )

        if close:
            for sym in all_symptoms:
                if sym.replace("_", " ") == close[0]:
                    extracted.append(sym)

    return list(set(extracted))

# ------------------ Prediction ------------------
def predict_disease(symptoms_list):
    input_vector = np.zeros(len(symptoms_dict))

    for symptom in symptoms_list:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1

    input_frame = pd.DataFrame([input_vector], columns=cols)

    pred_class = model.predict(input_frame)[0]
    pred_proba = model.predict_proba(input_frame)[0]

    disease = le.inverse_transform([pred_class])[0]
    confidence = round(pred_proba[pred_class] * 100, 2)

    return disease, confidence, pred_proba

# ------------------ Quotes ------------------
quotes = [
    "🌸 Health is wealth, take care of yourself.",
    "💪 A healthy outside starts from the inside.",
    "☀️ Every day is a chance to get stronger and healthier.",
    "🌿 Take a deep breath, your health matters the most.",
    "🌺 Remember, self-care is not selfish."
]

# ------------------ Chatbot ------------------
def chatbot():
    getSeverityDict()
    getDescription()
    getprecautionDict()

    print("\n======================================")
    print("🚀 XGBoost Model Trained")
    print(f"📊 Training Accuracy : {train_acc:.2f}%")
    print(f"📊 Testing Accuracy  : {test_acc:.2f}%")
    print(f"🎯 Precision         : {precision:.2f}%")
    print(f"📢 Recall            : {recall:.2f}%")
    print(f"🏆 F1-score          : {f1:.2f}%")
    print("======================================\n")

    print("🤖 Welcome to HealthCare ChatBot")

    name = input("👉 What is your name? : ")
    age = input("👉 Please enter your age: ")
    gender = input("👉 What is your gender? (M/F/Other): ")

    symptoms_input = input("👉 Describe your symptoms: ")

    symptoms_list = extract_symptoms(symptoms_input, cols)

    if not symptoms_list:
        print("❌ Sorry, no valid symptoms found.")
        return

    print("✅ Detected symptoms:", ", ".join(symptoms_list))

    input("👉 For how many days? : ")
    input("👉 Severity 1-10: ")
    input("👉 Any pre-existing conditions? : ")
    input("👉 Smoking/drinking/irregular sleep? : ")
    input("👉 Family history? : ")

    disease, confidence, proba = predict_disease(symptoms_list)

    print("\n----------- RESULT -----------")
    print(f"🩺 Disease Prediction : {disease}")
    print(f"🔎 Confidence         : {confidence}%")
    print(f"📖 About : {description_list.get(disease, 'No description available.')}")

    if disease in precautionDictionary:
        print("\n🛡️ Suggested precautions:")
        for i, prec in enumerate(precautionDictionary[disease], 1):
            print(f"{i}. {prec}")

    print("\n💡 " + random.choice(quotes))
    print(f"\nThank you {name}!")

# ------------------ Run ------------------
if __name__ == "__main__":
    chatbot()