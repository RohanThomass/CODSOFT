from pathlib import Path
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from imblearn.over_sampling import SMOTE

# =====================================
# Project Paths
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"

MODEL_DIR = BASE_DIR / "models"

FIGURE_DIR = BASE_DIR / "outputs" / "figures"

REPORT_DIR = BASE_DIR / "outputs" / "reports"

MODEL_DIR.mkdir(exist_ok=True)

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# Create folders if they don't exist
# ==========================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

# ==========================================
# Load Dataset
# ==========================================

print("="*60)
print("Loading Dataset...")
print("="*60)

df = pd.read_csv("data/raw/creditcard.csv")

print(df.head())
print("\nDataset Shape:", df.shape)

# ==========================================
# Features & Target
# ==========================================

X = df.drop("Class", axis=1)
y = df["Class"]

# ==========================================
# Scale Time & Amount
# ==========================================

scaler = StandardScaler()

X[["Time", "Amount"]] = scaler.fit_transform(
    X[["Time", "Amount"]]
)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Set:", X_train.shape)
print("Testing Set :", X_test.shape)

# ==========================================
# SMOTE
# ==========================================

print("\nApplying SMOTE...")

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)

print("After SMOTE")
print(y_train.value_counts())

# ==========================================
# Model
# ==========================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training Completed!")

# ==========================================
# Prediction
# ==========================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# ==========================================
# Evaluation
# ==========================================

accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
roc = roc_auc_score(y_test,y_prob)

print("\n")
print("="*60)
print("MODEL PERFORMANCE")
print("="*60)

print(classification_report(y_test,y_pred))

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc:.4f}")

# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test,y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap="Blues"
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/figures/confusion_matrix.png")

plt.close()

# ==========================================
# Save Model
# ==========================================

print("\nSaving Model...")

joblib.dump(
    model,
    "models/fraud_detector.pkl"
)

print("Model Saved Successfully!")

# ==========================================
# Save Report
# ==========================================

results = pd.DataFrame({

    "Metric":[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC"
    ],

    "Value":[
        accuracy,
        precision,
        recall,
        f1,
        roc
    ]

})

results.to_csv(
    "outputs/reports/model_results.csv",
    index=False
)

print("\nEverything Completed Successfully!")
