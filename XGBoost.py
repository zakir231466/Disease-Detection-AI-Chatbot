import re
import random
import pandas as pd
import numpy as np
import csv
from pathlib import Path
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from difflib import get_close_matches
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ------------------ Load Data ------------------
def resolve_base_dir():
    script_dir = Path(__file__).resolve().parent
    candidates = [script_dir, script_dir.parent]
    for candidate in candidates:
        if (candidate / "Data").exists() and (candidate / "Master Data").exists():
            return candidate
    raise FileNotFoundError("Data folders not found")

BASE_DIR = resolve_base_dir()

training = pd.read_csv(BASE_DIR / "Data" / "Training.csv")
testing  = pd.read_csv(BASE_DIR / "Data" / "Testing.csv")

# Clean columns
training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
testing.columns = testing.columns.str.replace(r"\.\d+$", "", regex=True)

training = training.loc[:, ~training.columns.duplicated()]
testing = testing.loc[:, ~testing.columns.duplicated()]

# Features & labels
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

# Encode labels
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42
)

# ------------------ XGBoost Model ------------------
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)

model.fit(x_train, y_train)

# ------------------ Dictionaries ------------------
severityDictionary = {}
description_list = {}
precautionDictionary = {}
symptoms_dict = {symptom: idx for idx, symptom in enumerate(x)}

def getDescription():
    with open(BASE_DIR / "Master Data" / "symptom_Description.csv") as f:
        for row in csv.reader(f):
            description_list[row[0]] = row[1]

def getSeverityDict():
    with open(BASE_DIR / "Master Data" / "Symptom_severity.csv") as f:
        for row in csv.reader(f):
            try:
                severityDictionary[row[0]] = int(row[1])
            except:
                pass

def getprecautionDict():
    with open(BASE_DIR / "Master Data" / "symptom_precaution.csv") as f:
        for row in csv.reader(f):
            precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

# ------------------ Symptom Extraction ------------------
symptom_synonyms = {
    "stomach ache": "stomach_pain",
    "belly pain": "stomach_pain",
    "tummy pain": "stomach_pain",
    "loose motion": "diarrhea",
    "motions": "diarrhea",
    "high temperature": "fever",
    "feaver": "fever",
    "coughing": "cough",
    "throat pain": "sore_throat",
    "shortness of breath": "breathlessness",
    "body ache": "muscle_pain",
}

def extract_symptoms(user_input, all_symptoms):
    extracted = []
    text = user_input.lower()

    for phrase, mapped in symptom_synonyms.items():
        if phrase in text:
            extracted.append(mapped)

    for symptom in all_symptoms:
        if symptom.replace("_", " ") in text:
            extracted.append(symptom)

    words = re.findall(r"\w+", text)
    for word in words:
        close = get_close_matches(word, [s.replace("_", " ") for s in all_symptoms], cutoff=0.8)
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

    input_vector = np.array(input_vector).reshape(1, -1)

    pred_proba = model.predict_proba(input_vector)[0]
    pred_class = np.argmax(pred_proba)

    disease = le.inverse_transform([pred_class])[0]
    confidence = round(pred_proba[pred_class] * 100, 2)

    return disease, confidence

# ------------------ Chatbot ------------------
def chatbot():
    getSeverityDict()
    getDescription()
    getprecautionDict()

    print("🤖 Welcome to XGBoost HealthCare ChatBot")

    name = input("Name: ")
    age = input("Age: ")
    gender = input("Gender: ")

    symptoms_input = input("Describe symptoms: ")
    symptoms_list = extract_symptoms(symptoms_input, cols)

    if not symptoms_list:
        print("No symptoms detected.")
        return

    print("Detected:", symptoms_list)

    num_days = input("Days: ")
    severity_scale = input("Severity (1-10): ")

    disease, confidence = predict_disease(symptoms_list)

    print("\n------ RESULT ------")
    print("Disease:", disease)
    print("Confidence:", confidence, "%")

    print("Description:", description_list.get(disease, "N/A"))

    if disease in precautionDictionary:
        print("\nPrecautions:")
        for p in precautionDictionary[disease]:
            print("-", p)

    print("\nTake care,", name)

# ------------------ Run ------------------
if __name__ == "__main__":
    chatbot()