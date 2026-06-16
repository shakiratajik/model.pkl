"""
BACKEND
-------
This file is the "brain" of the project.
It loads the data, cleans it, trains the Machine Learning model,
and saves the trained model to a file (model.pkl).

The frontend (app.py) will simply LOAD this saved file and use it.
We don't want to retrain the model every time someone opens the website
-- that would be slow. So we train ONCE here, save it, and reuse it.

HOW TO RUN THIS FILE:
    python backend.py

After running, you will see a new file created: model.pkl
That file contains your trained model + scaler + encoders, all packed together.
"""

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# =========================
# STEP 1: Load Dataset
# =========================
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
print("Data loaded:", df.shape)

# =========================
# STEP 2: Data Cleaning
# =========================
df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.fillna(df.median(numeric_only=True), inplace=True)

# =========================
# STEP 3: Encode text columns into numbers
# =========================
# We must remember HOW each column was encoded, so the frontend
# can encode new user input the SAME way. So we save one encoder
# PER COLUMN in a dictionary instead of reusing a single encoder.
encoders = {}

for col in df.columns:
    if df[col].dtype == "object" or df[col].dtype == "string" or pd.api.types.is_string_dtype(df[col]):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le   # remember this column's encoder

# =========================
# STEP 4: Split features (X) and target (y)
# =========================
X = df.drop("Churn", axis=1)
y = df["Churn"]

feature_columns = X.columns.tolist()  # remember the exact column order

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# STEP 5: Scale numeric features
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# STEP 6: Train the model
# =========================
# Random Forest is chosen as the final model because it generally gives
# the best accuracy among the models you tested (LR, DT, RF, KNN, NB).
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# =========================
# STEP 7: Evaluate the model
# =========================
pred = model.predict(X_test_scaled)

print("\nModel Performance on Test Data:")
print("Accuracy :", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall   :", recall_score(y_test, pred))
print("F1 Score :", f1_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

# =========================
# STEP 8: Save everything needed for prediction later
# =========================
# We bundle the model, the scaler, the encoders, and the column order
# into ONE file so the frontend can load it in a single line.
bundle = {
    "model": model,
    "scaler": scaler,
    "encoders": encoders,
    "feature_columns": feature_columns,
}

with open("model.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("\nSaved trained model to model.pkl")
print("You can now run the frontend with: streamlit run app.py")
