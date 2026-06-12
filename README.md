# 🏥 Disease Detection AI Chatbot

An AI-powered disease prediction system built with Python. 
The project ships in **two modes**: a **web application** (Flask + browser UI) and a **terminal chatbot**.
Both accept a patient's natural-language symptom description and return a predicted disease, confidence score, description, 
and recommended precautions — powered by one of five selectable ML models.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Two Modes of Use](#two-modes-of-use)
- [Available ML Models](#available-ml-models)
- [How It Works](#how-it-works)
- [Model Performance](#model-performance)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Web App — app.py](#web-app--apppy)
  - [Terminal Chatbot — test.py](#terminal-chatbot--testpy)
  - [Standalone Model Scripts](#standalone-model-scripts)
  - [Logistic Regression Evaluation Tool](#logistic-regression-evaluation-tool)
- [API Reference](#api-reference)
- [Symptom Synonyms Supported](#symptom-synonyms-supported)
- [Outputs Generated](#outputs-generated)
- [Limitations](#limitations)
- [Disclaimer](#disclaimer)

---

## Project Overview

This system predicts diseases from binary symptom data across **41 disease classes** using **132 symptom features**.
A patient describes their symptoms in plain text; the system extracts relevant symptoms using synonym mapping, exact matching, and
fuzzy matching, then feeds a binary feature vector into a trained ML classifier to return the most likely diagnosis.

The project supports five different classification algorithms, allowing direct comparison of their performance on the same dataset.

---

## Project Structure

```
Disease-Detection-AI-Chatbot-Test/
│
├── app.py                        # Flask web application (all 5 models, REST API)
├── test.py                       # Terminal chatbot (model selection at runtime)
│
├── random_forest.py              # Standalone RF script — fixed, with evaluation report
├── decision_tree.py              # Standalone DT script — saves .pkl + tree diagram
├── svm.py                        # Standalone SVM script — saves confusion matrix
├── xgboost_model.py              # Standalone XGBoost script — saves evaluation txt
├── logistic_regression.py        # Logistic Regression evaluation tool (CLI flags)
│
├── requirements.txt              # Python dependencies
│
├── Data/
│   ├── Training.csv              # 4,920 rows x 133 cols (132 symptoms + prognosis)
│   ├── Testing.csv               # 41 rows — 1 per disease (see note below)
│   └── dataset.csv               # Alternate format used by SVM and Decision Tree
│
├── Master Data/
│   ├── symptom_Description.csv   # Disease -> plain-English description
│   ├── Symptom_severity.csv      # Symptom -> severity weight (1–7)
│   └── symptom_precaution.csv    # Disease -> 4 precautionary actions
│
├── models/
│   ├── random_forest.pkl         # Pre-trained Random Forest model
│   ├── decision_tree.pkl         # Pre-trained Decision Tree model
│   ├── logistic_regression.pkl   # Pre-trained Logistic Regression model
│   ├── svm.pkl                   # Pre-trained SVM model
│   ├── xgboost.pkl               # Pre-trained XGBoost model
│   └── features.txt              # Feature list used during training
│
└── templates/
    └── index.html                # Full single-page web UI
```

---

## Two Modes of Use

### Web Application (`app.py`)
A browser-based chatbot powered by Flask. All five models are trained at startup and accessible via a dropdown. The patient fills in their details, describes symptoms, answers yes/no follow-up questions, and receives a visual diagnosis report — all without leaving the browser.

### Terminal Chatbot (`test.py`)
A command-line version of the same chatbot. On launch, the user selects one of five models. The session runs interactively in the terminal, asking questions and printing results as formatted text.

---

## Available ML Models

| # | Model | Script | Key Config |
|---|-------|--------|-----------|
| 1 | Random Forest | `random_forest.py` | 300 trees, max_depth=12, regularised |
| 2 | Decision Tree | `decision_tree.py` | criterion=entropy, max_depth=15 |
| 3 | Logistic Regression | `logistic_regression.py` | max_iter=1000, C=1.0 |
| 4 | Support Vector Machine | `svm.py` | kernel=linear, C=1.0 |
| 5 | XGBoost | `xgboost_model.py` | 300 estimators, lr=0.1, max_depth=6 |

All five models are also available simultaneously inside `app.py` and `test.py`.

---

## How It Works

```
Patient describes symptoms in plain text
              |
              v
     +---------------------+
     |   Symptom Extractor  |
     |  1. Synonym mapping  |  "stomach ache" -> stomach_pain
     |  2. Exact match      |  "fever" -> fever
     |  3. Fuzzy match      |  "nauseu" -> nausea  (cutoff 0.80)
     +---------------------+
              |
              v
   Build binary feature vector
   (132 dimensions: 1=present, 0=absent)
              |
              v
   ML model initial prediction
              |
              v
   Guided follow-up questions
   (up to 8 yes/no questions based
    on the initially predicted disease)
              |
              v
   Final ML prediction
              |
              v
   Output: Disease  Confidence %
           Description  Precautions
           Severity score




## Dataset

### Training.csv
- **4,920 rows**, 133 columns (132 binary symptom features + `prognosis` label)
- **41 disease classes**, exactly 120 rows per disease
- After `drop_duplicates()`: **303 unique rows** — each row is repeated ~16x
- Features are binary: `1` = symptom present, `0` = symptom absent

### Testing.csv
- 41 rows — one per disease
- **All 41 rows are exact copies from Training.csv** (100% overlap confirmed). It is not a genuine hold-out set. `random_forest.py` ignores it and uses an internal stratified 80/20 split instead. Other scripts that use this file will show inflated accuracy.

### dataset.csv
- Alternative format used by `svm.py` and `decision_tree.py`
- Stores symptoms as categorical strings (e.g. `"fever"`, `"cough"`) rather than binary columns
- Converted to numerical severity weights during preprocessing using `Symptom_severity.csv`

### Master Data

| File | Contents |
|------|----------|
| `symptom_Description.csv` | Maps each disease to a one-sentence plain-English description |
| `Symptom_severity.csv` | Maps each symptom to a severity weight (integer 1–7) |
| `symptom_precaution.csv` | Maps each disease to 4 recommended precautionary actions |

---

## Installation

### Requirements

- Python 3.8+
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:

```
flask
numpy
pandas
scikit-learn
xgboost
matplotlib
joblib
seaborn
```

### Folder structure

Ensure the folder structure is preserved exactly as shown in 
[Project Structure](#project-structure). The scripts auto-detect the `Data` and `Master Data`
folders relative to their own location — do not rename or move them.

---

## Usage

### Web App — `app.py`

```bash
python app.py
```

Then open your browser at:

```
http://127.0.0.1:5000
```

**What happens:**
1. All five models are trained at startup (~30–60 seconds depending on hardware)
2. A chat interface opens in the browser
3. Select a model from the dropdown
4. Enter patient name, age, gender
5. Describe symptoms in a sentence
6. Answer yes/no follow-up questions
7. Receive the full diagnosis report

---

### Terminal Chatbot — `test.py`

```bash
python test.py
```

**Model selection menu shown on startup:**

```
==================================================
Choose a Model:
==================================================
1. Random Forest
2. Decision Tree
3. Logistic Regression
4. Support Vector Machine (SVM)
5. XGBoost
==================================================
Enter model number (1-5):
```

After selecting a model, the chatbot session begins. Example session:

```
HealthCare ChatBot (Random Forest)
Hello! Please answer a few questions so I can understand your condition better.

What is your name? : Sarah
Please enter your age: 28
What is your gender? (M/F/Other): F
Describe your symptoms in a sentence: I have high fever and yellowish skin

Detected symptoms: fever, yellowing_of_eyes

For how many days have you had these symptoms? : 4
On a scale of 1-10, how severe do you feel? : 6
Any pre-existing conditions? : None
Do you smoke, drink, or have irregular sleep? : No
Any family history of similar illness? : No

Let me ask a few more questions related to Jaundice...
Do you also have vomiting? (yes/no): yes
Do you also have fatigue? (yes/no): yes

================ RESULT ================
Predicted Disease : Jaundice
Confidence        : 87.5%
About : Jaundice is a condition in which the skin turns yellow...

Suggested precautions:
   1. drink plenty of water
   2. consume milk thistle
   3. eat fruits and high fibre food
   4. medication
```

---

### Standalone Model Scripts

Each model has its own standalone training and evaluation script.

#### `random_forest.py` — Corrected RF with honest metrics
```bash
python random_forest.py
```
Prints full evaluation report (accuracy, CV, precision, recall, F1, per-class breakdown) then launches the chatbot.

#### `decision_tree.py` — Decision Tree with visual export
```bash
python decision_tree.py
```
Saves to disk:
- `decision_tree_model.pkl` — trained model
- `symptom_features.pkl` — feature list
- `decision_tree_graph.png` — visual flowchart of the top 3 tree levels

#### `svm.py` — SVM with confusion matrix
```bash
python svm.py
```
Saves to disk:
- `evaluation_report.txt` — accuracy, precision, recall, F1, classification report
- `confusion_matrix.png` — 41x41 heatmap of predictions vs actuals

#### `xgboost_model.py` — XGBoost with cross-validation
```bash
python xgboost_model.py
```
Saves to disk:
- `xgboost_evaluation_report.txt` — full metrics including 5-fold CV scores and confusion matrix

---

### Logistic Regression Evaluation Tool

`logistic_regression.py` is a dedicated CLI evaluation tool with configurable flags:

```bash
# Default: run full evaluation
python logistic_regression.py

# Check overlap between Training.csv and Testing.csv
python logistic_regression.py --overlap

# Check row diversity / duplicate analysis
python logistic_regression.py --diversity

# Custom noise rate and regularisation
python logistic_regression.py --evaluate --flip-rate 0.10 --C 1.0

# Save output to a timestamped file
python logistic_regression.py --save --output-dir reports
```

**Available flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--overlap` | off | Show how many test rows exist verbatim in training |
| `--diversity` | off | Show unique row counts per disease class |
| `--evaluate` | on | Run full model evaluation |
| `--flip-rate` | 0.10 | Fraction of test features to randomly flip (noise simulation) |
| `--C` | 1.0 | Logistic Regression regularisation strength |
| `--save` | off | Save terminal output to a timestamped `.txt` file |
| `--output-dir` | reports | Folder to write saved report files into |

---

## API Reference

`app.py` exposes two REST endpoints consumed by the web frontend.

### `GET /api/models`

Returns train and test accuracy for all five models.

**Response:**
```json
{
  "models": {
    "logistic_regression": { "train": 100.0, "test": 97.5 },
    "random_forest":        { "train": 100.0, "test": 100.0 },
    "svm":                  { "train": 100.0, "test": 100.0 },
    "decision_tree":        { "train": 100.0, "test": 100.0 },
    "xgboost":              { "train": 100.0, "test": 100.0 }
  }
}
```

---

### `POST /api/chat`

Drives the multi-step conversation. Each request advances the session by one step and returns the next bot message(s).

**Request body fields:**

| Field | Type | Description |
|-------|------|-------------|
| `step` | string | Current step: `welcome`, `symptoms`, or `refinement` |
| `patient` | object | `{ name, age, gender }` |
| `model_name` | string | `random_forest`, `decision_tree`, `logistic_regression`, `svm`, or `xgboost` |
| `user_input` | string | The patient's text for this step |
| `symptoms` | array | Accumulated confirmed symptoms so far |
| `symptoms_to_ask` | array | Remaining follow-up symptom questions |
| `current_question_idx` | int | Index into `symptoms_to_ask` |
| `predicted_disease` | string | Disease from the initial prediction step |

**Conversation flow:**

```
welcome  -->  symptoms  -->  refinement (x0 to 8)  -->  complete
```

**Response when `step` is `complete` includes a `result` object:**

```json
{
  "step": "complete",
  "result": {
    "disease":     "Jaundice",
    "confidence":  87.5,
    "description": "Jaundice is a condition where the skin turns yellow...",
    "precautions": ["drink plenty of water", "consume milk thistle", "eat fruits and high fibre food", "medication"],
    "severity":    3.2,
    "quote":       "Health is wealth, take care of yourself."
  }
}
```

---

## Symptom Synonyms Supported

The system maps common everyday phrasings to the correct internal feature names:

| You type | Mapped to |
|---|---|
| stomach ache / belly pain / tummy pain | `stomach_pain` |
| loose motion / motions | `diarrhea` |
| high temperature / temperature / feaver | `fever` |
| coughing | `cough` |
| throat pain | `sore_throat` |
| cold | `chills` |
| breathing issue / shortness of breath | `breathlessness` |
| body ache | `muscle_pain` |

Beyond synonyms, **fuzzy matching** (similarity cutoff: 0.80) handles minor typos and misspellings automatically.

---

## Outputs Generated

| Script | Files saved to disk |
|--------|---------------------|
| `decision_tree.py` | `decision_tree_model.pkl`, `symptom_features.pkl`, `decision_tree_graph.png` |
| `svm.py` | `evaluation_report.txt`, `confusion_matrix.png` |
| `xgboost_model.py` | `xgboost_evaluation_report.txt` |
| `logistic_regression.py` (with `--save`) | `reports/evaluation_YYYYMMDD_HHMMSS.txt` |

---

## Limitations

- **Synthetic dataset** — training data is computer-generated with perfectly unique, noise-free symptom patterns per disease. Real-world clinical performance will be lower.
- **Small effective sample size** — after deduplication, only ~303 unique rows exist across 41 classes (~7 per class). Per-class metrics are statistically fragile.
- **Testing.csv leakage** — the provided Testing.csv is a direct subset of Training.csv. Scripts that use it as a test set report inflated accuracy.
- **Binary features only** — symptom severity, duration, and demographics collected during the chat are not passed to the ML models.
- **No co-morbidity support** — predicts one disease at a time; cannot detect multiple simultaneous conditions.
- **41 fixed disease classes** — cannot predict conditions outside its training set.
- **Flask development server** — `app.py` runs with `debug=True`, intended for local use only, not production deployment.

---

## Disclaimer

> **This system is a decision-support tool for educational purposes only — not a medical diagnosis system.** Never use it as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for any health concerns. Do not delay seeking medical attention based on this chatbot's output.
