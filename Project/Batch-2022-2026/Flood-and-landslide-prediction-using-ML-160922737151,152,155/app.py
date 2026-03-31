import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = "static/model/flood/best_flood_model.pkl"
SCALER_PATH = "static/model/flood/scaler.pkl"
METRICS_PATH = "static/model/flood/performance_metrics.json"

MODEL_FOLDER = "static/model/landslide"
EDA_FOLDER = "static/eda/landslide"
PERF_FOLDER = "static/performance/landslide"

# =========================
# ROOT ROUTES
# =========================

@app.route("/")
def home():
    return render_template("home.html")

# =========================
# FLOOD ROUTES
# =========================

@app.route("/flood/eda")
def flood_eda():
    images = os.listdir("static/eda/flood")
    return render_template("flood/flood_eda.html", images=images)

@app.route("/flood/comparison")
def flood_comparison():

    performance_dir = "static/performance/flood"

    images = [f for f in os.listdir(performance_dir) if f.endswith(".png")]
    report_files = [f for f in os.listdir(performance_dir) if f.endswith(".txt")]

    reports = {}

    for file in report_files:
        with open(os.path.join(performance_dir, file), "r") as f:
            reports[file] = f.read()

    return render_template(
        "flood/flood_comparison.html",
        images=images,
        reports=reports
    )


@app.route("/flood/model")
def flood_model():
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    return render_template("flood/flood_model.html", metrics=metrics)

@app.route("/flood/predict", methods=["GET", "POST"])
def flood_predict():

    prediction = None
    error = None

    if request.method == "POST":

        import pandas as pd

        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        with open("static/model/flood/feature_names.json") as f:
            feature_names = json.load(f)

        try:
            input_data = {}

            for feature in feature_names:
                value = request.form.get(feature)

                if value is None or value == "":
                    raise ValueError(f"Missing input for {feature}")

                input_data[feature] = float(value)

            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)

            result = model.predict(input_scaled)[0]

            prediction = "Flood Likely" if result == 1 else "No Flood"

        except Exception as e:
            error = str(e)

    return render_template(
        "flood/flood_prediction.html",
        prediction=prediction,
        error=error
    )

# ---------------------------
# 1️⃣ Landslide EDA
# ---------------------------
@app.route("/landslide/eda")
def landslide_eda():

    images = os.listdir(EDA_FOLDER)

    distributions = [img for img in images if "distribution" in img]
    boxplots = [img for img in images if "boxplot" in img]
    heatmaps = [img for img in images if "heatmap" in img]
    others = [img for img in images if img not in distributions + boxplots + heatmaps]

    return render_template("landslide/landslide_eda.html",
                           distributions=distributions,
                           boxplots=boxplots,
                           heatmaps=heatmaps,
                           others=others)


# ---------------------------
# 2️⃣ Model Comparison
# ---------------------------
@app.route("/landslide/comparison")
def landslide_comparison():

    performance_path = os.path.join("static", "performance", "landslide")

    images = []
    reports = {}

    for file in os.listdir(performance_path):

        if file.endswith(".png"):
            images.append(file)

        if file.endswith(".txt"):  # classification reports
            with open(os.path.join(performance_path, file), "r", encoding="utf-8") as f:
                reports[file] = f.read()

    return render_template(
        "landslide/landslide_comparison.html",
        images=images,
        reports=reports
    )

# ---------------------------
# 3️⃣ Final Model Details
# ---------------------------
@app.route("/landslide/model")
def landslide_model():

    with open(f"{MODEL_FOLDER}/performance_metrics.json") as f:
        metrics = json.load(f)

    with open(f"{MODEL_FOLDER}/classification_report.txt") as f:
        report = f.read()

    with open(f"{MODEL_FOLDER}/feature_columns.json") as f:
        columns = json.load(f)

    return render_template("landslide/landslide_model.html",
                           metrics=metrics,
                           report=report,
                           columns=columns)


# ---------------------------
# 4️⃣ Prediction
# ---------------------------
@app.route("/landslide/predict", methods=["GET", "POST"])
def landslide_predict():

    prediction = None
    confidence = None
    error = None

    if request.method == "POST":
        try:
            model = joblib.load(f"{MODEL_FOLDER}/best_model.pkl")
            scaler = joblib.load(f"{MODEL_FOLDER}/scaler.pkl")
            label_encoder = joblib.load(f"{MODEL_FOLDER}/label_encoder.pkl")

            with open(f"{MODEL_FOLDER}/feature_columns.json") as f:
                columns = json.load(f)

            input_data = {}

            for col in columns:
                value = request.form.get(col)

                if value is None or value == "":
                    input_data[col] = 0
                else:
                    input_data[col] = float(value)

            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)

            # Predict class
            result = model.predict(input_scaled)[0]

            # Get class probabilities
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_scaled)[0]
                confidence = round(np.max(probabilities) * 100, 2)

            prediction = label_encoder.inverse_transform([result])[0]

            #prediction = str(result)

        except Exception as e:
            error = str(e)

    # Load columns for dynamic form
    with open(f"{MODEL_FOLDER}/feature_columns.json") as f:
        columns = json.load(f)

    return render_template("landslide/landslide_prediction.html",
                           columns=columns,
                           prediction=prediction,
                           confidence=confidence,
                           error=error)


if __name__ == "__main__":
    app.run(debug=True)
