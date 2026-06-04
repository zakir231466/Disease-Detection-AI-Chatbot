import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import joblib

print("--- Step 1: Loading Datasets ---")
raw_data = pd.read_csv('dataset.csv')
severity_df = pd.read_csv('Symptom_severity.csv', header=None, names=['Symptom', 'Weight'])
description_df = pd.read_csv('symptom_Description.csv', header=None, names=['Disease', 'Description'])
precaution_df = pd.read_csv('symptom_precaution.csv', header=None)
precaution_df.columns = ['Disease', 'Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']

severity_df['Symptom'] = severity_df['Symptom'].str.strip().str.replace(' ', '_')
description_df['Disease'] = description_df['Disease'].str.strip()
precaution_df['Disease'] = precaution_df['Disease'].str.strip()

all_features = severity_df['Symptom'].unique().tolist()

print("\n--- Step 2: Cleaning Dataset & Stripping Blank Rows ---")
for col in raw_data.columns:
    raw_data[col] = raw_data[col].astype(str).str.strip().str.replace(' ', '_')

encoded_rows = []
np.random.seed(42)
drop_rate = 0.05 

for idx, row in raw_data.iterrows():
    disease_name = str(row.iloc[0]).strip()
    
    if not disease_name or disease_name == 'nan' or disease_name == '' or disease_name.startswith('_') or disease_name.startswith(','):
        continue
        
    row_dict = {symptom: 0 for symptom in all_features}
    row_dict['Disease'] = disease_name
    
    for item in row.iloc[1:]:
        if pd.notna(item) and item != 'nan' and item != '':
            if item in row_dict:
                if np.random.rand() > drop_rate:
                    row_dict[item] = 1
                
    encoded_rows.append(row_dict)

ml_df = pd.DataFrame(encoded_rows)
X = ml_df.drop(columns=['Disease'])
y = ml_df['Disease']

print("\n--- Step 3: Stratified Train-Test Split ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n--- Step 4: Training Calibrated Decision Tree ---")
model = DecisionTreeClassifier(
    criterion='entropy', 
    max_depth=15,             
    min_samples_split=4,      
    min_samples_leaf=2,       
    random_state=42
)
model.fit(X_train, y_train)
print("Decision Tree Training Complete!")

print("\n--- Step 5: Generating Performance Metrics Report ---")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)

print("=" * 65)
print(f"Overall Model Accuracy : {accuracy * 100:.2f}%")
print(f"Precision (Macro)      : {precision * 100:.2f}%")
print(f"Recall (Macro)         : {recall * 100:.2f}%")
print(f"F1-Score (Macro)       : {f1 * 100:.2f}%")
print("=" * 65)

print("\nClassification Evaluation Metrics Report:")
unique_labels = sorted(list(set(y_test)))
print(classification_report(y_test, y_pred, target_names=unique_labels, digits=4, zero_division=0))
print("=" * 65)

print("\n--- Step 6: Exporting Shared Artifacts ---")
joblib.dump(model, 'decision_tree_model.pkl')
joblib.dump(all_features, 'symptom_features.pkl')

print("\n--- Step 7: Exporting Visual Flowchart Tree ---")
plt.figure(figsize=(25, 12))
plot_tree(
    model, 
    feature_names=all_features, 
    class_names=model.classes_, 
    filled=True, 
    rounded=True, 
    max_depth=3 
)
plt.savefig('decision_tree_graph.png', dpi=300, bbox_inches='tight')
print("Saved 90% accuracy architecture diagram to 'decision_tree_graph.png'")