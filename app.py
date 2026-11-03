"""Streamlit app: evaluate pre-trained purchase-intent classifiers on an uploaded test CSV."""
import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
from data_utils import FEATURES, TARGET, prepare

MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
LABELS = ["No purchase (0)", "Purchase (1)"]

st.set_page_config(page_title="Shopper Intent - Model Evaluator", layout="wide")


@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[name]))


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


st.title("Online Shoppers Purchase Intention - Model Evaluator")
st.write("Upload a **test CSV** containing the 17 session features and the `Revenue` "
         "column (1/True = purchase, 0/False = no purchase). The saved models are "
         "loaded and evaluated on it - nothing is retrained.")

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test CSV", type=["csv"])
    model_name = st.selectbox("Select model", list(MODEL_FILES.keys()))
    st.caption("Required columns: " + ", ".join(FEATURES + [TARGET]))

if uploaded is None:
    st.info("Upload a CSV in the sidebar to begin. Use `test_data.csv` from the repository.")
    st.stop()

raw = pd.read_csv(uploaded)
missing = [c for c in FEATURES + [TARGET] if c not in raw.columns]
if missing:
    st.error("Uploaded file is missing required columns: " + ", ".join(missing))
    st.stop()

raw = raw.dropna(subset=FEATURES + [TARGET])
try:
    X, y = prepare(raw)
except Exception as exc:  # malformed values
    st.error(f"Could not read the uploaded data: {exc}")
    st.stop()

if y.nunique() < 2:
    st.error("The Revenue column must contain both classes to compute AUC.")
    st.stop()

st.subheader("Uploaded data preview")
st.write(f"{X.shape[0]} rows, {X.shape[1]} features")
st.dataframe(raw.head(10), width="stretch")

model = load_model(model_name)
pred, metrics = compute_metrics(model, X, y)

st.subheader(f"Evaluation metrics - {model_name}")
for col, (k, v) in zip(st.columns(6), metrics.items()):
    col.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=LABELS, yticklabels=LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.subheader("Classification report")
    rep = classification_report(y, pred, target_names=LABELS, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).transpose().round(4), width="stretch")

st.subheader("All models on the uploaded data")
rows = []
for name in MODEL_FILES:
    _, m = compute_metrics(load_model(name), X, y)
    rows.append({"ML Model Name": name, **m})
st.dataframe(pd.DataFrame(rows).round(4), width="stretch", hide_index=True)
