from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

app = Flask(__name__)

# ======================================================
# Paths
# ======================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "outputs"
IMAGE_FOLDER = BASE_DIR / "static" / "images"

UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)
IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)

# ======================================================
# Load Model
# ======================================================

model = joblib.load(BASE_DIR / "models" / "fraud_detector.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")

# ======================================================
# Home
# ======================================================

@app.route("/")
def home():
    return render_template("index.html")


# ======================================================
# Predict
# ======================================================

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No file uploaded."

    file = request.files["file"]

    if file.filename == "":
        return "Please select a CSV file."

    filepath = UPLOAD_FOLDER / file.filename
    file.save(filepath)

    df = pd.read_csv(filepath)

    has_actual = False
    metrics = {}

    # -----------------------------
    # Separate features and target
    # -----------------------------

    if "Class" in df.columns:
        has_actual = True
        y_true = df["Class"]
        X = df.drop(columns=["Class"])
    else:
        X = df.copy()

    # -----------------------------
    # Scale features
    # -----------------------------

    X[["Time", "Amount"]] = scaler.transform(X[["Time", "Amount"]])

    # -----------------------------
    # Prediction
    # -----------------------------

    predictions = model.predict(X)

    X["Prediction"] = predictions
    X["Prediction"] = X["Prediction"].map({
        0: "Genuine",
        1: "Fraud"
    })

    # Save prediction column to original dataframe
    df["Prediction"] = X["Prediction"]

    # -----------------------------
    # Metrics (if Class exists)
    # -----------------------------

    if has_actual:

        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions)
        recall = recall_score(y_true, predictions)
        f1 = f1_score(y_true, predictions)

        probabilities = model.predict_proba(X.drop(columns=["Prediction"]))[:, 1]

        roc = roc_auc_score(y_true, probabilities)

        metrics = {
            "accuracy": round(accuracy,4),
            "precision": round(precision,4),
            "recall": round(recall,4),
            "f1": round(f1,4),
            "roc": round(roc,4)
        }

        cm = confusion_matrix(y_true, predictions)

        plt.figure(figsize=(5,4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues"
        )

        plt.title("Confusion Matrix")

        plt.savefig(IMAGE_FOLDER / "confusion_matrix.png")

        plt.close()

    # -----------------------------
    # Summary
    # -----------------------------

    fraud_count = (df["Prediction"] == "Fraud").sum()
    genuine_count = (df["Prediction"] == "Genuine").sum()

    # -----------------------------
    # Pie Chart
    # -----------------------------

    plt.figure(figsize=(5,5))

    plt.pie(
        [genuine_count, fraud_count],
        labels=["Genuine", "Fraud"],
        autopct="%1.1f%%",
        colors=["green", "red"],
        startangle=90
    )

    plt.title("Prediction Distribution")

    plt.savefig(IMAGE_FOLDER / "pie_chart.png")

    plt.close()

    # -----------------------------
    # Bar Chart
    # -----------------------------

    plt.figure(figsize=(5,4))

    plt.bar(
        ["Genuine", "Fraud"],
        [genuine_count, fraud_count],
        color=["green", "red"]
    )

    plt.title("Prediction Summary")

    plt.savefig(IMAGE_FOLDER / "bar_chart.png")

    plt.close()

    # -----------------------------
    # Save CSV
    # -----------------------------

    output_file = OUTPUT_FOLDER / "predictions.csv"

    df.to_csv(output_file, index=False)

    # -----------------------------
    # Render
    # -----------------------------

    return render_template(
        "result.html",
        total=len(df),
        fraud=fraud_count,
        genuine=genuine_count,
        metrics=metrics,
        has_actual=has_actual,
        table=df.head(20).to_html(
            classes="table",
            index=False
        )
    )


# ======================================================
# Download
# ======================================================

@app.route("/download")
def download():

    return send_file(
        OUTPUT_FOLDER / "predictions.csv",
        as_attachment=True
    )


# ======================================================
# Run
# ======================================================

if __name__ == "__main__":
    app.run(debug=True)