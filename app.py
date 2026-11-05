"""Streamlit app: evaluate pre-trained purchase-intent classifiers on an uploaded test CSV."""

import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from data_utils import FEATURES, TARGET, prepare

# Import evaluation metrics from Scikit-learn to measure model performance.
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Define the directory and filenames of the five pre-trained machine learning models.
MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

# Define class labels for purchase and no-purchase predictions.
LABELS = ["No purchase (0)", "Purchase (1)"]

# Configure the Streamlit page layout and title.
st.set_page_config(page_title="Shopper Intent - Model Evaluator", layout="wide")

# Load the selected pre-trained model and cache it to avoid repeated file loading.
@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[name]))

# Generate predictions and calculate six performance metrics for the selected model.
def compute_metrics(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return pred, {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, proba),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }


# Display the application title and instructions for uploading the test dataset.
st.title("Online Shoppers Purchase Intention - Model Evaluator")
st.write(
    "Upload a **test CSV** containing the 17 session features and the `Revenue` "
    "column (1/True = purchase, 0/False = no purchase). The saved models are "
    "loaded and evaluated on it - nothing is retrained."
)


# Provide controls for uploading the test CSV and selecting a pre-trained model.
with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test CSV", type=["csv"])
    model_name = st.selectbox("Select model", list(MODEL_FILES.keys()))
    st.caption("Required columns: " + ", ".join(FEATURES + [TARGET]))

# Stop execution if the user has not uploaded a test dataset.
if uploaded is None:
    st.info(
        "Upload a CSV in the sidebar to begin. Use `test_data.csv` from the repository."
    )
    st.stop()

# Read the uploaded CSV and validate the presence of all required feature and target columns.
raw = pd.read_csv(uploaded)
missing = [c for c in FEATURES + [TARGET] if c not in raw.columns]
if missing:
    st.error("Uploaded file is missing required columns: " + ", ".join(missing))
    st.stop()

# Remove incomplete records and preprocess the dataset into input features (X) and target (y).
raw = raw.dropna(subset=FEATURES + [TARGET])
try:
    X, y = prepare(raw)
except Exception as exc:  # malformed values
    st.error(f"Could not read the uploaded data: {exc}")
    st.stop()

# Ensure both purchase and no-purchase classes exist for reliable AUC evaluation.
if y.nunique() < 2:
    st.error("The Revenue column must contain both classes to compute AUC.")
    st.stop()

# Display a preview of the uploaded dataset and its dimensions.
st.subheader("Uploaded data preview")
st.write(f"{X.shape[0]} rows, {X.shape[1]} features")
st.dataframe(raw.head(10), width="stretch")

# Load the selected model and evaluate its performance on the uploaded test dataset.
model = load_model(model_name)
pred, metrics = compute_metrics(model, X, y)

# Display the six evaluation metrics of the selected machine learning model.
st.subheader(f"Evaluation metrics - {model_name}")
for col, (k, v) in zip(st.columns(6), metrics.items()):
    col.metric(k, f"{v:.4f}")


# Divide the interface into two sections for displaying evaluation visualizations.
left, right = st.columns(2)
with left:
      # Generate and visualize the confusion matrix to compare actual and predicted classes.
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=LABELS,
        yticklabels=LABELS,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
        # Generate and display the classification report containing precision, recall, F1-score, and support.
    st.subheader("Classification report")
    rep = classification_report(
        y, pred, target_names=LABELS, output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(rep).transpose().round(4), width="stretch")

# Evaluate all five pre-trained models on the same test dataset for performance comparison.
st.subheader("All models on the uploaded data")
rows = []
for name in MODEL_FILES:
    _, m = compute_metrics(load_model(name), X, y)
    rows.append({"ML Model Name": name, **m})
# Display the comparative evaluation results of all models in a single table.
st.dataframe(pd.DataFrame(rows).round(4), width="stretch", hide_index=True)
