import re
import random
import pandas as pd
import numpy as np
import csv
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__, template_folder='templates', static_folder='static')

# ------------------ Load and Train Models ------------------
def resolve_base_dir():
    """Find the folder that contains both Data and Master Data."""
    script_dir = Path(__file__).resolve().parent
    candidates = [script_dir, script_dir.parent]
    for candidate in candidates:
        if (candidate / "Data").exists() and (candidate / "Master Data").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find 'Data' and 'Master Data' folders next to the script or one level above."
    )

BASE_DIR = resolve_base_dir()
training = pd.read_csv(BASE_DIR / "Data" / "Training.csv")
testing  = pd.read_csv(BASE_DIR / "Data" / "Testing.csv")

# Clean duplicate column names
training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
testing.columns = testing.columns.str.replace(r"\.\d+$", "", regex=True)
training = training.loc[:, ~training.columns.duplicated()]
testing = testing.loc[:, ~testing.columns.duplicated()]

# Features and labels
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

# Encode target
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Train all models at startup
print("Training models...")
trained_models = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "svm": SVC(kernel='linear', probability=True, random_state=42),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "xgboost": XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
}

model_accuracies = {}
for name, m in trained_models.items():
    m.fit(x_train, y_train)
    train_acc = m.score(x_train, y_train) * 100
    test_acc = m.score(x_test, y_test) * 100
    model_accuracies[name] = {
        "train": round(train_acc, 2),
        "test": round(test_acc, 2)
    }
    print(f"Loaded {name} model. Test Accuracy: {test_acc:.2f}%")

# Load Dictionaries
severityDictionary = {}
description_list = {}
precautionDictionary = {}
symptoms_dict = {symptom: idx for idx, symptom in enumerate(cols)}

def load_dictionaries():
    with open(BASE_DIR / "Master Data" / "symptom_Description.csv") as csv_file:
        for row in csv.reader(csv_file):
            if len(row) >= 2:
                description_list[row[0]] = row[1]

    with open(BASE_DIR / "Master Data" / "Symptom_severity.csv") as csv_file:
        for row in csv.reader(csv_file):
            if len(row) >= 2:
                try:
                    severityDictionary[row[0]] = int(row[1])
                except:
                    pass

    with open(BASE_DIR / "Master Data" / "symptom_precaution.csv") as csv_file:
        for row in csv.reader(csv_file):
            if len(row) >= 5:
                precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

load_dictionaries()

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

def extract_symptoms(user_input, all_symptoms):
    extracted = []
    text = user_input.lower().replace("-", " ")

    # 1. Synonym replacement
    for phrase, mapped in symptom_synonyms.items():
        if phrase in text:
            extracted.append(mapped)

    # 2. Exact match
    for symptom in all_symptoms:
        if symptom.replace("_", " ") in text:
            extracted.append(symptom)

    # 3. Fuzzy match (typo handling)
    words = re.findall(r"\w+", text)
    for word in words:
        close = get_close_matches_api(word, [s.replace("_", " ") for s in all_symptoms], n=1, cutoff=0.8)
        if close:
            for sym in all_symptoms:
                if sym.replace("_", " ") == close[0]:
                    extracted.append(sym)

    return list(set(extracted))

def get_close_matches_api(word, possibilities, n=1, cutoff=0.8):
    from difflib import get_close_matches
    return get_close_matches(word, possibilities, n=n, cutoff=cutoff)

def predict_disease(symptoms_list, model_name):
    model = trained_models.get(model_name, trained_models["random_forest"])
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms_list:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1

    pred_proba = model.predict_proba([input_vector])[0]
    pred_class = np.argmax(pred_proba)
    disease = le.inverse_transform([pred_class])[0]
    confidence = round(pred_proba[pred_class] * 100, 2)
    return disease, confidence

# Empathy Quotes
quotes = [
    "🌸 Health is wealth, take care of yourself.",
    "💪 A healthy outside starts from the inside.",
    "☀️ Every day is a chance to get stronger and healthier.",
    "🌿 Take a deep breath, your health matters the most.",
    "🌺 Remember, self-care is not selfish."
]

# ------------------ Routes ------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/models", methods=["GET"])
def get_models():
    return jsonify({
        "models": model_accuracies
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    step = data.get("step", "welcome")
    patient = data.get("patient", {})
    model_name = data.get("model_name", "random_forest")
    symptoms = data.get("symptoms", [])
    symptoms_to_ask = data.get("symptoms_to_ask", [])
    current_question_idx = data.get("current_question_idx", 0)
    user_input = data.get("user_input", "").strip()
    predicted_disease = data.get("predicted_disease", "")

    if step == "welcome":
        # Check input details
        name = patient.get("name", "User")
        return jsonify({
            "step": "symptoms",
            "patient": patient,
            "model_name": model_name,
            "symptoms": [],
            "symptoms_to_ask": [],
            "current_question_idx": 0,
            "bot_messages": [
                f"👋 Hello {name}! I am your AI Health Assistant.",
                "Describe your symptoms in a simple sentence (e.g. 'I have high fever and stomach pain')."
            ],
            "suggested_actions": []
        })

    elif step == "symptoms":
        if not user_input:
            return jsonify({
                "step": "symptoms",
                "patient": patient,
                "model_name": model_name,
                "symptoms": [],
                "symptoms_to_ask": [],
                "current_question_idx": 0,
                "bot_messages": ["Please describe your symptoms so I can help you."],
                "suggested_actions": []
            })

        extracted = extract_symptoms(user_input, cols)
        if not extracted:
            return jsonify({
                "step": "symptoms",
                "patient": patient,
                "model_name": model_name,
                "symptoms": [],
                "symptoms_to_ask": [],
                "current_question_idx": 0,
                "bot_messages": [
                    "❌ I couldn't identify specific symptoms in that description.",
                    "Please try describing them in a different way or specify symptoms like 'headache', 'fever', or 'vomiting'."
                ],
                "suggested_actions": []
            })

        # Calculate initial prediction to get related symptoms
        disease, confidence = predict_disease(extracted, model_name)

        # Get other symptoms related to this predicted disease
        disease_symptoms = list(training[training['prognosis'] == disease].iloc[0][:-1].index[
            training[training['prognosis'] == disease].iloc[0][:-1] == 1
        ])

        # Filter out symptoms already reported
        symptoms_to_ask = [s for s in disease_symptoms if s not in extracted][:8]

        detected_formatted = ", ".join([s.replace("_", " ").title() for s in extracted])

        if not symptoms_to_ask:
            # Complete immediately if there are no follow-up symptoms
            final_disease, final_confidence = predict_disease(extracted, model_name)
            precautions = precautionDictionary.get(final_disease, ["Consult a doctor", "Rest well", "Stay hydrated"])
            desc = description_list.get(final_disease, "No description available.")
            
            # Calculate severity score
            total_severity = sum([severityDictionary.get(s, 1) for s in extracted])
            avg_severity = round(total_severity / len(extracted), 1) if extracted else 0
            
            return jsonify({
                "step": "complete",
                "patient": patient,
                "model_name": model_name,
                "symptoms": extracted,
                "symptoms_to_ask": [],
                "current_question_idx": 0,
                "bot_messages": [
                    f"✅ Detected symptoms: **{detected_formatted}**",
                    "Diagnosis completed! Please review your report below."
                ],
                "suggested_actions": [],
                "result": {
                    "disease": final_disease,
                    "confidence": final_confidence,
                    "description": desc,
                    "precautions": precautions,
                    "quote": random.choice(quotes),
                    "severity": avg_severity
                }
            })

        next_symptom = symptoms_to_ask[0].replace("_", " ").title()
        return jsonify({
            "step": "refinement",
            "patient": patient,
            "model_name": model_name,
            "symptoms": extracted,
            "symptoms_to_ask": symptoms_to_ask,
            "current_question_idx": 0,
            "predicted_disease": disease,
            "bot_messages": [
                f"✅ Detected symptoms: **{detected_formatted}**",
                f"Based on these, let me ask you some more questions related to potential **{disease}**.",
                f"👉 Do you also have **{next_symptom}**?"
            ],
            "suggested_actions": ["Yes", "No"]
        })

    elif step == "refinement":
        ans = user_input.strip().lower()
        if ans == "yes":
            symptoms.append(symptoms_to_ask[current_question_idx])

        next_idx = current_question_idx + 1

        if next_idx < len(symptoms_to_ask):
            next_symptom = symptoms_to_ask[next_idx].replace("_", " ").title()
            return jsonify({
                "step": "refinement",
                "patient": patient,
                "model_name": model_name,
                "symptoms": symptoms,
                "symptoms_to_ask": symptoms_to_ask,
                "current_question_idx": next_idx,
                "predicted_disease": predicted_disease,
                "bot_messages": [
                    f"👉 Do you also have **{next_symptom}**?"
                ],
                "suggested_actions": ["Yes", "No"]
            })
        else:
            # Diagnostics finished
            final_disease, final_confidence = predict_disease(symptoms, model_name)
            precautions = precautionDictionary.get(final_disease, ["Consult a doctor", "Rest well", "Stay hydrated"])
            desc = description_list.get(final_disease, "No description available.")

            # Calculate severity score
            total_severity = sum([severityDictionary.get(s, 1) for s in symptoms])
            avg_severity = round(total_severity / len(symptoms), 1) if symptoms else 0

            return jsonify({
                "step": "complete",
                "patient": patient,
                "model_name": model_name,
                "symptoms": symptoms,
                "symptoms_to_ask": [],
                "current_question_idx": 0,
                "bot_messages": [
                    "Thank you for answering my questions.",
                    "Diagnosis completed! Please review your report below."
                ],
                "suggested_actions": [],
                "result": {
                    "disease": final_disease,
                    "confidence": final_confidence,
                    "description": desc,
                    "precautions": precautions,
                    "quote": random.choice(quotes),
                    "severity": avg_severity
                }
            })

    return jsonify({"error": "Invalid step"}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
