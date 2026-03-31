import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ---------------------------------------
# CONFIG
# ---------------------------------------

DATA_PATH = "../landslide_risk_dataset.csv"
SAVE_PATH = "../static/model/landslide"

os.makedirs(SAVE_PATH, exist_ok=True)

# ---------------------------------------
# LOAD DATA
# ---------------------------------------

df = pd.read_csv(DATA_PATH)

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Rename target
df.rename(columns={
    "Landslide_Risk_Prediction": "target"
}, inplace=True)

if "target" not in df.columns:
    raise Exception("Target column not found.")

# ---------------------------------------
# HANDLE MISSING VALUES
# ---------------------------------------

for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# ---------------------------------------
# SPLIT FEATURES & TARGET
# ---------------------------------------

X = df.drop("target", axis=1)
y = df["target"]

# Encode target
if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)
    joblib.dump(le, f"{SAVE_PATH}/label_encoder.pkl")

# Handle categorical features
X = pd.get_dummies(X, drop_first=True)

# Save feature columns
feature_columns = X.columns.tolist()

with open(f"{SAVE_PATH}/feature_columns.json", "w") as f:
    json.dump(feature_columns, f)

# ---------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------
# HANDLE IMBALANCED DATA (SMOTE)
# ---------------------------------------

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# ---------------------------------------
# SCALE FEATURES
# ---------------------------------------

scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, f"{SAVE_PATH}/scaler.pkl")

# ---------------------------------------
# DEFINE MODELS
# ---------------------------------------

models = {
    "Random_Forest": RandomForestClassifier(),
    "Gradient_Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(eval_metric="logloss")
}

best_model = None
best_name = None
best_accuracy = 0
best_report = None
best_cm = None

# ---------------------------------------
# TRAIN + SELECT BEST
# ---------------------------------------

for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)

    print(f"{name} Accuracy:", acc)

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_name = name
        best_report = classification_report(y_test, y_pred)
        best_cm = confusion_matrix(y_test, y_pred)

# ---------------------------------------
# SAVE BEST MODEL
# ---------------------------------------

joblib.dump(best_model, f"{SAVE_PATH}/best_model.pkl")

# ---------------------------------------
# SAVE PERFORMANCE METRICS
# ---------------------------------------

metrics = {
    "best_model": best_name,
    "accuracy": float(best_accuracy)
}

with open(f"{SAVE_PATH}/performance_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Save classification report
with open(f"{SAVE_PATH}/classification_report.txt", "w") as f:
    f.write(best_report)

# Save confusion matrix image
plt.figure(figsize=(5,4))
sns.heatmap(best_cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"{best_name} Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{SAVE_PATH}/confusion_matrix.png")
plt.close()

# ---------------------------------------
# SAVE 4 TRUE POSITIVE SAMPLES PER CLASS
# ---------------------------------------

# Predict again using best model
y_test_pred = best_model.predict(X_test_scaled)

# Convert to DataFrame for easy handling
X_test_df = pd.DataFrame(X_test, columns=feature_columns)
X_test_df.reset_index(drop=True, inplace=True)

y_test_series = pd.Series(y_test).reset_index(drop=True)
y_pred_series = pd.Series(y_test_pred)

# Combine into single DataFrame
results_df = X_test_df.copy()
results_df["Actual"] = y_test_series
results_df["Predicted"] = y_pred_series

# Get unique classes
unique_classes = sorted(results_df["Actual"].unique())

tp_samples = []

for cls in unique_classes:

    # True Positives for this class
    tp_class = results_df[
        (results_df["Actual"] == cls) &
        (results_df["Predicted"] == cls)
    ]

    # Take first 4 (or less if not enough)
    tp_class_4 = tp_class.head(4)

    tp_samples.append(tp_class_4)

# Combine all classes
final_tp_df = pd.concat(tp_samples)

# If label encoder exists, decode labels
if os.path.exists(f"{SAVE_PATH}/label_encoder.pkl"):
    le = joblib.load(f"{SAVE_PATH}/label_encoder.pkl")
    final_tp_df["Actual_Label"] = le.inverse_transform(final_tp_df["Actual"])
    final_tp_df["Predicted_Label"] = le.inverse_transform(final_tp_df["Predicted"])

# Save to CSV
final_tp_df.to_csv(f"{SAVE_PATH}/true_positive_samples.csv", index=False)

print("True Positive samples saved successfully.")

print("Best Model:", best_name)
print("Accuracy:", best_accuracy)
print("All artifacts saved in:", SAVE_PATH)
