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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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
# STEP 6: Train multiple models and compare them
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
}

results = []
trained_models = {}

for name, clf in models.items():
    clf.fit(X_train_scaled, y_train)
    pred = clf.predict(X_test_scaled)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1 Score": f1_score(y_test, pred),
    })
    trained_models[name] = clf

results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
print("\nModel Comparison:")
print(results_df)

# The best model (highest accuracy) becomes the one used for live predictions
best_model_name = results_df.iloc[0]["Model"]
model = trained_models[best_model_name]
print(f"\nBest model selected for predictions: {best_model_name}")

pred = model.predict(X_test_scaled)

# =========================
# STEP 7: Evaluate the chosen best model in detail
# =========================
print(f"\n{best_model_name} Performance on Test Data:")
print("Accuracy :", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall   :", recall_score(y_test, pred))
print("F1 Score :", f1_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

# =========================
# STEP 8: Prepare EDA summary stats for the Overview page
# =========================
# We use the ORIGINAL (pre-encoding) data for human-readable charts.
df_raw = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df_raw["TotalCharges"] = pd.to_numeric(df_raw["TotalCharges"], errors="coerce")
df_raw.fillna(df_raw.median(numeric_only=True), inplace=True)

eda_stats = {
    "total_customers": len(df_raw),
    "churn_rate": (df_raw["Churn"] == "Yes").mean() * 100,
    "avg_tenure": df_raw["tenure"].mean(),
    "avg_monthly_charges": df_raw["MonthlyCharges"].mean(),
    "churn_counts": df_raw["Churn"].value_counts().to_dict(),
    "contract_churn": pd.crosstab(df_raw["Contract"], df_raw["Churn"]).to_dict(),
}

# =========================
# STEP 9: Customer Segmentation (K-Means Clustering)
# =========================
# We group customers into segments based on their behaviour
# (tenure, monthly charges, total charges) so the business can
# target different groups differently (e.g. "high spend, short tenure").
X_cluster = df_raw[["tenure", "MonthlyCharges", "TotalCharges"]].copy()

scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_cluster_scaled)

df_raw["Segment"] = cluster_labels

# Reduce to 2D for a simple scatter plot in the frontend
pca = PCA(n_components=2, random_state=42)
cluster_2d = pca.fit_transform(X_cluster_scaled)

segmentation_data = pd.DataFrame({
    "PCA1": cluster_2d[:, 0],
    "PCA2": cluster_2d[:, 1],
    "Segment": cluster_labels.astype(str),
})

# Average profile of each segment - helps explain what each group means
segment_profile = df_raw.groupby("Segment")[["tenure", "MonthlyCharges", "TotalCharges"]].mean().round(1)
segment_profile["Customer Count"] = df_raw["Segment"].value_counts().sort_index()
segment_profile["Churn Rate (%)"] = (
    df_raw.groupby("Segment")["Churn"].apply(lambda x: (x == "Yes").mean() * 100).round(1)
)

print("\nCustomer Segments:")
print(segment_profile)

# =========================
# STEP 10: Save everything needed by the frontend
# =========================
# We bundle the model, the scaler, the encoders, and the column order
# into ONE file so the frontend can load it in a single line.
bundle = {
    "model": model,
    "best_model_name": best_model_name,
    "scaler": scaler,
    "encoders": encoders,
    "feature_columns": feature_columns,
    "results_df": results_df,
    "eda_stats": eda_stats,
    "segmentation_data": segmentation_data,
    "segment_profile": segment_profile,
}

with open("model.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("\nSaved trained model to model.pkl")
print("You can now run the frontend with: streamlit run app.py")
