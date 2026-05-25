

# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt                  
import seaborn as sns                            
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import warnings                      
warnings.filterwarnings("ignore")

def load_and_preprocess_data():
    """
    Reads the raw CSV files, cleans the text, and maps textual symptoms 
    to their numerical severity values.
    """
    # 1. Load datasets
    df = pd.read_csv('dataset.csv')
    severity_df = pd.read_csv('Symptom_severity.csv', names=['Symptom', 'Weight'])
    
    # 2. Clean the symptom strings and create a mapping dictionary
    severity_df['Symptom'] = severity_df['Symptom'].str.strip().str.replace(' ', '_')
    severity_dict = dict(zip(severity_df['Symptom'], severity_df['Weight']))
    
    # 3. Clean the Main Dataset
    df = df.fillna('0')
    target_col = df.columns[0] 
    symptom_cols = df.columns[1:]
    
    # Standardize string formatting in symptom columns
    for col in symptom_cols:
        df[col] = df[col].apply(lambda x: str(x).strip().replace(' ', '_') if isinstance(x, str) else x)
        
    # 4. Feature Transformation (Categorical to Numerical)
    X = df[symptom_cols].copy()
    for col in symptom_cols:
        X[col] = X[col].map(severity_dict)
        X[col] = X[col].fillna(0) # Unmapped values become 0
        
    y = df[target_col]
    
    return X, y

def main():
    print("Loading and preprocessing the dataset...")
    X, y = load_and_preprocess_data()
    
    # --- Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- Model Initialization and Training ---
    svm_model = SVC(kernel='linear', C=1.0, random_state=42)
    
    print("Training the Support Vector Machine (SVM) model. Please wait...")
    svm_model.fit(X_train, y_train)
    
    # --- Making Predictions ---
    y_pred = svm_model.predict(X_test)
    
    # --- Model Evaluation Calculations ---
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    # Generate the classification report and confusion matrix
    class_report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # ==========================================
    # 1. SHOW EVALUATION IN TERMINAL
    # ==========================================
    print("\n" + "="*50)
    print("        SVM MODEL EVALUATION REPORT")
    print("="*50)
    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1-score  : {f1 * 100:.2f}%\n")
    print("-" * 50)
    print("DETAILED CLASSIFICATION REPORT")
    print("-" * 50)
    print(class_report)
    
    # ==========================================
    # 2. SAVE EVALUATION REPORT TO FILE
    # ==========================================
    with open('evaluation_report.txt', 'w') as file:
        
        file.write("        SVM MODEL EVALUATION REPORT\n")
        file.write("="*50 + "\n\n")
        file.write(f"Accuracy  : {accuracy * 100:.2f}%\n")
        file.write(f"Precision : {precision * 100:.2f}%\n")
        file.write(f"Recall    : {recall * 100:.2f}%\n")
        file.write(f"F1-score  : {f1 * 100:.2f}%\n\n")
        file.write("-" * 50 + "\n")
        file.write("DETAILED CLASSIFICATION REPORT\n")
        file.write("-" * 50 + "\n")
        file.write(class_report)
    print("-> Saved: 'evaluation_report.txt' created in the folder.")
        
    # ==========================================
    # 3. SAVE AND SHOW CONFUSION MATRIX IMAGE
    # ==========================================
    print("-> Generating confusion matrix...")
    plt.figure(figsize=(24, 20)) 
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=svm_model.classes_, 
                yticklabels=svm_model.classes_)
    
    plt.title('SVM Disease Prediction Confusion Matrix', fontsize=24, pad=20)
    plt.xlabel('Predicted Disease', fontsize=18, labelpad=15)
    plt.ylabel('Actual Disease', fontsize=18, labelpad=15)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    # Save the file FIRST
    plt.savefig('confusion_matrix.png', dpi=300) 
    print("-> Saved: 'confusion_matrix.png' created in the folder.")
    print("\nProcess complete! Opening Confusion Matrix Viewer...")

    # Display it on the screen LAST (This will pause the script until you close the image window)
    plt.show() 

if __name__ == "__main__":
    main()