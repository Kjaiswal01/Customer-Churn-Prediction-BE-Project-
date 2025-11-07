# model_training.py
"""
End-to-end pipeline for Customer Churn Prediction using ML + NLP
Includes EDA, baseline RandomForest, hyperparameter tuning, and artifact saving.
"""

import os
import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from imblearn.over_sampling import SMOTE
import joblib

# ---------------------------
# 0. Config & paths
# ---------------------------
DATA_PATH = "data/customer_churn.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
RANDOM_STATE = 42

# ---------------------------
# 1. Load dataset
# ---------------------------
print("Loading dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Rows:", df.shape[0], "Cols:", df.shape[1])

# ---------------------------
# 2. Basic Data Cleaning
# ---------------------------
df.dropna(how='all', inplace=True)
required_cols = ["Customer_ID","Gender","Age","Tenure","Subscription_Type",
                 "Monthly_Charges","Total_Spend","Last_Interaction_Days","Feedback","Churn"]
for c in required_cols:
    if c not in df.columns:
        raise Exception(f"Required column missing: {c}")

df = df.dropna(subset=["Gender","Age","Tenure","Monthly_Charges","Total_Spend","Feedback","Churn"])

# ---------------------------
# 3. EDA (basic)
# ---------------------------
print("\n=== Dataset Info ===")
print(df.info())
print("\n=== Statistical Summary ===")
print(df.describe())
print("\n=== Churn Value Counts ===")
print(df['Churn'].value_counts())

# Gender distribution plot
plt.figure(figsize=(5,4))
sns.countplot(x='Gender', data=df)
plt.title("Gender Distribution")
plt.savefig(os.path.join(MODELS_DIR, "eda_gender.png"))
plt.close()

# Subscription distribution plot
plt.figure(figsize=(6,4))
sns.countplot(x='Subscription_Type', data=df)
plt.title("Subscription Type Distribution")
plt.savefig(os.path.join(MODELS_DIR, "eda_subscription.png"))
plt.close()

# Churn vs Subscription
plt.figure(figsize=(6,4))
sns.countplot(x='Subscription_Type', hue='Churn', data=df)
plt.title("Churn by Subscription Type")
plt.savefig(os.path.join(MODELS_DIR, "eda_churn_subscription.png"))
plt.close()

# ---------------------------
# 4. NLP: Sentiment extraction (TextBlob)
# ---------------------------
def get_sentiment(text):
    try:
        return round(TextBlob(str(text)).sentiment.polarity, 3)
    except:
        return 0.0

df["Sentiment"] = df["Feedback"].apply(get_sentiment)

# ---------------------------
# 5. Encode target and categorical features
# ---------------------------
# Target: Churn -> 0/1
df['Churn'] = df['Churn'].map(lambda x: 1 if str(x).strip().lower() in ["yes","1","y","true","t"] else 0)

# Label encode categorical features
label_encoders = {}
cat_cols = ["Gender","Subscription_Type"]
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

joblib.dump(label_encoders, os.path.join(MODELS_DIR, "label_encoders.pkl"))

# ---------------------------
# 6. Feature selection & scaling
# ---------------------------
feature_cols = [
    "Gender", "Age", "Tenure", "Subscription_Type",
    "Monthly_Charges", "Total_Spend", "Last_Interaction_Days", "Sentiment"
]
X = df[feature_cols].copy()
y = df["Churn"].astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_list.pkl"))

# ---------------------------
# 7. Train-test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------
# 8. Handle class imbalance (SMOTE)
# ---------------------------
sm = SMOTE(random_state=RANDOM_STATE)
X_res, y_res = sm.fit_resample(X_train, y_train)

# ---------------------------
# 9. Baseline RandomForest
# ---------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced")
rf.fit(X_res, y_res)

# ---------------------------
# 10. Hyperparameter tuning (GridSearchCV)
# ---------------------------
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "class_weight": ["balanced", None]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=RANDOM_STATE),
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    scoring="f1",
    verbose=1
)
grid_search.fit(X_res, y_res)
best_rf = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

# ---------------------------
# 11. Evaluation on test set
# ---------------------------
y_pred = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:,1]

print("\n=== Tuned RandomForest Test Metrics ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "confusion_matrix.png"))
plt.close()

# Feature importance plot
importances = best_rf.feature_importances_
fi = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
plt.figure(figsize=(8,4))
sns.barplot(x=fi.values, y=fi.index)
plt.title("Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "feature_importance.png"))
plt.close()

# ---------------------------
# 12. Save tuned model
# ---------------------------
joblib.dump(best_rf, os.path.join(MODELS_DIR, "churn_model_tuned.pkl"))
print("Saved tuned model to models/churn_model_tuned.pkl")

# ---------------------------
# 13. Quick inference example
# ---------------------------
example_idx = 0
example_X = X_test[example_idx].reshape(1, -1)
pred = best_rf.predict(example_X)[0]
prob = best_rf.predict_proba(example_X)[0,1]
print(f"Example prediction (0=stay,1=churn): {pred} with prob {prob:.4f}")

print("\nAll done! Models and artifacts are saved in the 'models/' folder.")
