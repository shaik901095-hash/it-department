import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ==================================
# 1. Create Folder
# ==================================

model_path = "../static/model/flood"
os.makedirs(model_path, exist_ok=True)

# ==================================
# 2. Load Dataset
# ==================================

df = pd.read_csv("../kerala.csv")
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Drop constant column
if "SUBDIVISION" in df.columns:
    df.drop("SUBDIVISION", axis=1, inplace=True)

# Encode target
if df["FLOODS"].dtype == "object":
    le = LabelEncoder()
    df["FLOODS"] = le.fit_transform(df["FLOODS"])

# ==================================
# 3. Split Data
# ==================================

X = df.drop("FLOODS", axis=1)
y = df["FLOODS"]

feature_names = X.columns.tolist()

with open(f"{model_path}/feature_names.json", "w") as f:
    json.dump(feature_names, f)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================================
# 4. Train Strongest Models
# ==================================

models = {
    "Random_Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "Gradient_Boosting": GradientBoostingClassifier()
}

results = {}

for name, model in models.items():

    if name == "Random_Forest":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)

    results[name] = {
        "model": model,
        "accuracy": acc,
        "y_pred": y_pred
    }

# ==================================
# 5. Select Best Model
# ==================================

best_model_name = max(results, key=lambda x: results[x]["accuracy"])
best_model = results[best_model_name]["model"]
best_accuracy = results[best_model_name]["accuracy"]
y_pred_best = results[best_model_name]["y_pred"]

print(f"\nBest Model Selected: {best_model_name}")
print(f"Accuracy: {best_accuracy:.4f}")

# ==================================
# 6. Save Model & Scaler
# ==================================

joblib.dump(best_model, f"{model_path}/best_flood_model.pkl")
joblib.dump(scaler, f"{model_path}/scaler.pkl")

# ==================================
# 7. Save Performance Metrics
# ==================================

metrics = {
    "model_name": best_model_name,
    "accuracy": float(best_accuracy),
    "precision": float(precision_score(y_test, y_pred_best)),
    "recall": float(recall_score(y_test, y_pred_best)),
    "f1_score": float(f1_score(y_test, y_pred_best))
}

with open(f"{model_path}/performance_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Save Classification Report
report = classification_report(y_test, y_pred_best)

with open(f"{model_path}/classification_report.txt", "w") as f:
    f.write(report)

# ==================================
# 8. Save Confusion Matrix
# ==================================

cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title(f"{best_model_name} Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{model_path}/confusion_matrix.png")
plt.close()

print("\nModel and Performance Saved Successfully!")
print("Location:", model_path)
