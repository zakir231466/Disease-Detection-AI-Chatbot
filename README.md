🏥 Healthcare Chatbot — Disease Prediction System

A terminal-based AI chatbot that predicts diseases from user-described symptoms using 
a Random Forest classifier. The system collects patient information, extracts symptoms from natural language input, 
runs a guided diagnostic session, and returns a predicted disease with confidence score, description, and precautions.


📋 Table of Contents

Overview
Project Structure
Features
How It Works
Model Details
Performance Metrics
Dataset
Data Leakage — What Was Fixed
Installation
Usage
Sample Session
Symptom Synonyms Supported
Limitations
Disclaimer



Overview

The Healthcare Chatbot takes a patient's symptom description as plain text, 
maps it to 132 binary symptom features, and uses a trained Random Forest model
to predict the most likely disease from 41 possible conditions. After an initial prediction,
the chatbot asks targeted follow-up questions to refine the result, then outputs the disease name,
confidence percentage, a plain-English description, and suggested precautions.


Project Structure

Ai_chatbot/
│
├── Data/
│   ├── Training.csv          # 4,920 rows × 133 cols (132 symptoms + prognosis)
│   └── Testing.csv           # 41 rows — 1 per disease (not used; see note below)
│
├── Master Data/
│   ├── symptom_Description.csv   # Disease name → plain-English description
│   ├── Symptom_severity.csv      # Symptom name → severity score (1–7)
│   └── symptom_precaution.csv    # Disease name → 4 precautionary actions
│
└── test.py                   # Main script — model training + chatbot


Features


Natural language symptom input — type symptoms as a sentence, not a list
Synonym mapping — understands common variants like "stomach ache", "feaver", "shortness of breath"
Fuzzy matching — catches minor spelling mistakes using difflib
Guided follow-up questions — asks up to 8 targeted yes/no questions to refine the prediction
Confidence score — reports the probability of the predicted disease
Disease description — plain-English explanation of the predicted condition
Precaution list — up to 4 recommended actions for the predicted disease
Full evaluation report — prints accuracy, precision, recall, F1, CV score, and per-class breakdown on every run



How It Works

User types symptoms
        │
        ▼
Extract symptoms from text
  ├── Synonym mapping  ("stomach ache" → stomach_pain)
  ├── Exact match      ("fever" → fever)
  └── Fuzzy match      ("nauseu" → nausea)
        │
        ▼
Build binary feature vector (132 dimensions)
        │
        ▼
Random Forest initial prediction
        │
        ▼
Guided follow-up questions (up to 8)
        │
        ▼
Final prediction with confidence %
        │
        ▼
Output: Disease · Description · Precautions


Model Details

ParameterValueAlgorithmRandom Forest Classifiern_estimators300max_depth12min_
samples_split10min_samples_leaf5max_featuressqrtclass_weightbalancedrandom_state42Train
/test split80% / 20% stratifiedCross-validation5-fold StratifiedKFoldNoise injection7% 
random bit-flipLabel encodingsklearn LabelEncoder


Performance Metrics

Metrics are computed on a proper 20% stratified hold-out split after deduplication 
and noise injection. These are printed automatically every time the script runs.

MetricScoreTraining Accuracy100.00%Test Accuracy (20%)90.16%CV Accuracy (5-fold)85.18% ± 4.20%Precision 
(weighted)86.34%Recall (weighted)90.16%F1-score (weighted)87.60%Train − Test gap9.84% (moderate)


The training accuracy of 100% is expected for a Random Forest on structured data and 
does not indicate a problem. The meaningful score is the test accuracy (90.16%) and the
cross-validation score (85.18%), which measure real generalisation.




Dataset

Training.csv


4,920 rows, 133 columns (132 binary symptom columns + prognosis label)
41 disease classes, 120 rows per disease
After drop_duplicates(): 303 unique rows (each row was repeated ~16×)
All features are binary: 1 = symptom present, 0 = symptom absent


Testing.csv


41 rows — one per disease
Not used for evaluation — all 41 rows are exact copies of rows in Training.csv
(100% overlap), making it unsuitable as a genuine hold-out set. The script uses an internal stratified 80/20 split instead.


Master Data files

FileContentssymptom_Description.csvMaps disease name to a one-sentence descriptionSymptom_severity.
csvMaps symptom name to a severity weight (1–7)symptom_precaution.csvMaps disease name to 4 precautionary actions


Data Leakage — What Was Fixed

The original implementation reported 100% accuracy on both train and test sets. 
This was caused by four compounding issues:

1. Row Duplication

Training.csv contains 4,920 rows but only 303 are unique. Every row is repeated ~16 times. 
Without deduplication, identical rows appear in both the training and test partitions, 
so the model simply memorises every test case during training.

Fix: X_all.drop_duplicates() before splitting.

2. Contaminated Testing.csv

Every row in Testing.csv is an exact copy of a row already in Training.csv. 
Using it as a test set means the model has already seen every test sample — guaranteed 100% accuracy.

Fix: Ignore Testing.csv. Use a stratified 80/20 split from the deduplicated training data.

3. Synthetic Zero-Noise Dataset

The dataset is synthetically generated. Each of the 41 diseases has a perfectly unique,
noise-free binary symptom fingerprint. Any classifier — even a lookup table — achieves 
100% accuracy on this data without learning anything meaningful.

Fix: Apply 7% random bit-flip noise to simulate real-world symptom variability and force genuine generalisation.

4. Unlimited Tree Depth

Original hyperparameters (max_depth=None, min_samples_leaf=1) allowed trees to grow until every leaf contained a single pure sample — textbook memorisation.

Fix: max_depth=12, min_samples_leaf=5, min_samples_split=10.


Installation

Requirements


Python 3.8+
pip


Install dependencies

bashpip install pandas numpy scikit-learn

Clone / download

bash# Place the project folder so the structure matches:
Ai_chatbot/
├── Data/
│   ├── Training.csv
│   └── Testing.csv
├── Master Data/
│   ├── symptom_Description.csv
│   ├── Symptom_severity.csv
│   └── symptom_precaution.csv
└── test.py


Usage

bashcd Ai_chatbot
python test.py

The script will:


Train the Random Forest model (~5–10 seconds)
Print the full evaluation report
Start the interactive chatbot session



Sample Session

Training Random Forest model...

══════════════════════════════════════════════════════════
  🌲  RANDOM FOREST — MODEL EVALUATION REPORT
══════════════════════════════════════════════════════════

  DATASET SUMMARY
  ────────────────────────────────────────
  Original rows                  :   4920
  Unique rows (after dedup)      :    303
  Diseases / classes             :     41
  Train samples                  :    242
  Test  samples (20 %)           :     61

  CORE METRICS
  ────────────────────────────────────────
  Training Accuracy              : 100.00%
  Test Accuracy  (20 % hold-out) :  90.16%
  CV  Accuracy   (5-fold)        :  85.18%  ± 4.20%
  Precision  (weighted)          :  86.34%
  Recall     (weighted)          :  90.16%
  F1-score   (weighted)          :  87.60%

══════════════════════════════════════════════════════════

🤖 Welcome to HealthCare ChatBot

👉 What is your name? : Sarah
👉 Please enter your age: 28
👉 What is your gender? (M/F/Other): F

👉 Describe your symptoms in a sentence
   (e.g., 'I have fever and stomach pain'): I have high fever and yellowish skin

✅ Detected symptoms: fever, yellowing_of_eyes

👉 For how many days have you had these symptoms? : 4
👉 On a scale of 1–10, how severe do you feel? : 6
👉 Any pre-existing conditions (e.g., diabetes, hypertension)? : None
👉 Do you smoke, drink, or have irregular sleep? : No
👉 Any family history of similar illness? : No

🤔 Let me ask a few more questions related to Jaundice...
👉 Do you also have vomiting? (yes/no): yes
👉 Do you also have fatigue? (yes/no): yes
...

════════════════ RESULT ════════════════
🩺 Predicted Disease : Jaundice
🔎 Confidence        : 87.5%
📖 About : Jaundice is a condition where the skin and whites of the eyes turn yellow...

🛡️  Suggested precautions:
   1. drink plenty of water
   2. consume milk thistle
   3. eat fruits and high fibre food
   4. medication

💡 🌸 Health is wealth, take care of yourself.

Thank you for using the chatbot. Wishing you good health, Sarah!


Symptom Synonyms Supported

The chatbot understands common everyday phrasings and maps them to the correct feature names:

You typeMapped tostomach achestomach_painbelly painstomach_paintummy painstomach_painloose
motiondiarrheamotionsdiarrheahigh temperaturefevertemperaturefeverfeaverfevercoughingcoughthroat 
painsore_throatcoldchillsbreathing issuebreathlessnessshortness of breathbreathlessnessbody achemuscle_pain

In addition to synonyms, the chatbot performs fuzzy matching (cutoff: 0.80) to handle minor typos.


Limitations


Synthetic dataset — the training data is computer-generated and does not reflect real clinical distributions. Performance on real patients may be lower.
Small per-class test set — with only 1–2 test samples per disease, per-class metrics are statistically fragile. A single misclassification sets a class to 0% F1.
Binary symptom features only — symptom severity, duration, and demographic factors collected during the chat are not used by the model.
No co-morbidity modelling — the model predicts a single disease; it does not handle multiple simultaneous conditions.
41 fixed diseases — the model cannot predict conditions outside its training classes.



Disclaimer


This chatbot is a decision-support tool, not a medical diagnosis system. It is intended for educational and informational purposes only. Always consult a qualified healthcare professional for medical advice, diagnosis, or treatment. Do not rely on this tool for clinical decisions.
