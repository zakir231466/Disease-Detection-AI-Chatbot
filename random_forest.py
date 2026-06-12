import re
import random
import pandas as pd
import numpy as np
import csv
from pathlib import Path
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from difflib import get_close_matches
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ══════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════
def resolve_base_dir():
    script_dir = Path(__file__).resolve().parent
    for candidate in [script_dir, script_dir.parent]:
        if (candidate / "Data").exists() and (candidate / "Master Data").exists():
            return candidate
    raise FileNotFoundError("Could not find 'Data' and 'Master Data' folders.")

BASE_DIR = resolve_base_dir()
training = pd.read_csv(BASE_DIR / "Data" / "Training.csv")

training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
training = training.loc[:, ~training.columns.duplicated()]

cols  = training.columns[:-1]
X_all = training[cols]
y_all = training["prognosis"]

X_dedup = X_all.drop_duplicates().reset_index(drop=True)
y_dedup = y_all[X_all.drop_duplicates().index].reset_index(drop=True)


def add_noise(X_values, noise_rate=0.07, seed=42):
    rng    = np.random.default_rng(seed)
    X_noisy = X_values.astype(float).copy()
    mask   = rng.random(X_noisy.shape) < noise_rate
    X_noisy[mask] = 1.0 - X_noisy[mask]
    return X_noisy

X_noisy = add_noise(X_dedup.values, noise_rate=0.07)


le    = preprocessing.LabelEncoder()
y_enc = le.fit_transform(y_dedup)

X_train, X_test, y_train, y_test = train_test_split(
    X_noisy, y_enc,
    test_size=0.20,
    random_state=42,
    stratify=y_enc
)


model = RandomForestClassifier(
    n_estimators    = 300,
    max_depth       = 12,     # was None (unlimited)
    min_samples_split = 10,   # was 2
    min_samples_leaf  = 5,    # was 1
    max_features    = "sqrt",
    class_weight    = "balanced",
    random_state    = 42,
    n_jobs          = -1
)

print("Training Random Forest model...")
model.fit(X_train, y_train)

# ══════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════
y_train_pred = model.predict(X_train)
y_test_pred  = model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred) * 100
test_acc  = accuracy_score(y_test,  y_test_pred)  * 100
precision = precision_score(y_test, y_test_pred, average="weighted", zero_division=0) * 100
recall    = recall_score   (y_test, y_test_pred, average="weighted", zero_division=0) * 100
f1        = f1_score       (y_test, y_test_pred, average="weighted", zero_division=0) * 100

# 5-fold cross-validation
cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_noisy, y_enc, cv=cv, scoring="accuracy", n_jobs=-1)
cv_mean   = cv_scores.mean() * 100
cv_std    = cv_scores.std()  * 100

# Per-class classification report
class_report = classification_report(
    y_test, y_test_pred,
    target_names=le.classes_,
    zero_division=0
)

# ══════════════════════════════════════════════════════════════
#  PRINT RESULTS BANNER
# ══════════════════════════════════════════════════════════════
def print_results():
    W = 58
    sep = "═" * W

    print(f"\n{sep}")
    print("  🌲  RANDOM FOREST — MODEL EVALUATION REPORT")
    print(sep)

    # ── Dataset info ──────────────────────────────────────────
    print(f"\n  {'DATASET SUMMARY':}")
    print(f"  {'─'*40}")
    print(f"  {'Original rows':<30} : {len(X_all):>6}")
    print(f"  {'Unique rows (after dedup)':<30} : {len(X_dedup):>6}")
    print(f"  {'Diseases / classes':<30} : {len(le.classes_):>6}")
    print(f"  {'Train samples':<30} : {len(X_train):>6}")
    print(f"  {'Test  samples (20 %)':<30} : {len(X_test):>6}")

    # ── Core metrics ──────────────────────────────────────────
    print(f"\n  {'CORE METRICS':}")
    print(f"  {'─'*40}")
    print(f"  {'Training Accuracy':<30} : {train_acc:>6.2f}%")
    print(f"  {'Test Accuracy  (20 % hold-out)':<30} : {test_acc:>6.2f}%")
    print(f"  {'CV  Accuracy   (5-fold)':<30} : {cv_mean:>6.2f}%  ± {cv_std:.2f}%")
    print(f"  {'Precision  (weighted)':<30} : {precision:>6.2f}%")
    print(f"  {'Recall     (weighted)':<30} : {recall:>6.2f}%")
    print(f"  {'F1-score   (weighted)':<30} : {f1:>6.2f}%")

# ══════════════════════════════════════════════════════════════
#  DICTIONARIES
# ══════════════════════════════════════════════════════════════
severityDictionary   = {}
description_list     = {}
precautionDictionary = {}
symptoms_dict        = {symptom: idx for idx, symptom in enumerate(cols)}

def getDescription():
    with open(BASE_DIR / "Master Data" / "symptom_Description.csv") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                description_list[row[0]] = row[1]

def getSeverityDict():
    with open(BASE_DIR / "Master Data" / "Symptom_severity.csv") as f:
        for row in csv.reader(f):
            try:
                severityDictionary[row[0]] = int(row[1])
            except Exception:
                pass

def getprecautionDict():
    with open(BASE_DIR / "Master Data" / "symptom_precaution.csv") as f:
        for row in csv.reader(f):
            if len(row) >= 5:
                precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

# ══════════════════════════════════════════════════════════════
#  SYMPTOM SYNONYMS & EXTRACTOR
# ══════════════════════════════════════════════════════════════
symptom_synonyms = {
    "stomach ache":        "stomach_pain",
    "belly pain":          "stomach_pain",
    "tummy pain":          "stomach_pain",
    "loose motion":        "diarrhea",
    "motions":             "diarrhea",
    "high temperature":    "fever",
    "temperature":         "fever",
    "feaver":              "fever",
    "coughing":            "cough",
    "throat pain":         "sore_throat",
    "cold":                "chills",
    "breathing issue":     "breathlessness",
    "shortness of breath": "breathlessness",
    "body ache":           "muscle_pain",
}

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
            word, [s.replace("_", " ") for s in all_symptoms], n=1, cutoff=0.8
        )
        if close:
            for sym in all_symptoms:
                if sym.replace("_", " ") == close[0]:
                    extracted.append(sym)
    return list(set(extracted))

# ══════════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════════
def predict_disease(symptoms_list):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms_list:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
    pred_proba = model.predict_proba([input_vector])[0]
    pred_class = np.argmax(pred_proba)
    disease    = le.inverse_transform([pred_class])[0]
    confidence = round(pred_proba[pred_class] * 100, 2)
    return disease, confidence, pred_proba

# ══════════════════════════════════════════════════════════════
#  QUOTES
# ══════════════════════════════════════════════════════════════
quotes = [
    "🌸 Health is wealth, take care of yourself.",
    "💪 A healthy outside starts from the inside.",
    "☀️  Every day is a chance to get stronger and healthier.",
    "🌿 Take a deep breath, your health matters the most.",
    "🌺 Remember, self-care is not selfish.",
]

# ══════════════════════════════════════════════════════════════
#  CHATBOT
# ══════════════════════════════════════════════════════════════
def chatbot():
    getSeverityDict()
    getDescription()
    getprecautionDict()

    print_results()

    print("🤖 Welcome to HealthCare ChatBot")
    print("Hello! Please answer a few questions so I can understand your condition better.\n")

    name   = input("👉 What is your name? : ")
    age    = input("👉 Please enter your age: ")
    gender = input("👉 What is your gender? (M/F/Other): ")

    symptoms_input = input(
        "\n👉 Describe your symptoms in a sentence\n"
        "   (e.g., 'I have fever and stomach pain'): "
    )
    symptoms_list = extract_symptoms(symptoms_input, cols)

    if not symptoms_list:
        print("❌ Sorry, I could not detect valid symptoms. Please try again with more details.")
        return

    print(f"\n✅ Detected symptoms: {', '.join(symptoms_list)}")

    num_days       = int(input("\n👉 For how many days have you had these symptoms? : "))
    severity_scale = int(input("👉 On a scale of 1–10, how severe do you feel? : "))
    pre_exist      = input("👉 Any pre-existing conditions (e.g., diabetes, hypertension)? : ")
    lifestyle      = input("👉 Do you smoke, drink, or have irregular sleep? : ")
    family         = input("👉 Any family history of similar illness? : ")

    # Initial prediction
    disease, confidence, proba = predict_disease(symptoms_list)

    # Guided follow-up questions
    print(f"\n🤔 Let me ask a few more questions related to {disease}...")
    disease_rows = training[training["prognosis"] == disease]
    if not disease_rows.empty:
        disease_symptoms = list(
            disease_rows.iloc[0][:-1].index[disease_rows.iloc[0][:-1] == 1]
        )
        asked = 0
        for sym in disease_symptoms:
            if sym not in symptoms_list and asked < 8:
                ans = input(
                    f"👉 Do you also have {sym.replace('_', ' ')}? (yes/no): "
                ).strip().lower()
                if ans == "yes":
                    symptoms_list.append(sym)
                asked += 1

    # Final prediction
    disease, confidence, proba = predict_disease(symptoms_list)

    print("\n════════════════ RESULT ════════════════")
    print(f"🩺 Predicted Disease : {disease}")
    print(f"🔎 Confidence        : {confidence}%")
    print(f"📖 About : {description_list.get(disease, 'No description available.')}")

    if disease in precautionDictionary:
        print("\n🛡️  Suggested precautions:")
        for i, prec in enumerate(precautionDictionary[disease], 1):
            print(f"   {i}. {prec}")

    print("\n💡 " + random.choice(quotes))
    print(f"\nThank you for using the chatbot. Wishing you good health, {name}!")

# ══════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    chatbot()
