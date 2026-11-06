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

# Configure the Streamlit page layout, title, and icon.
st.set_page_config(
    page_title="Shopper Intent - Model Evaluator",
    page_icon="\U0001F6D2",
    layout="wide",
)

# Inject the external stylesheet, if present, for the app's visual styling.
_css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(_css_path):
    with open(_css_path) as _css_file:
        st.markdown(f"<style>{_css_file.read()}</style>", unsafe_allow_html=True)

# Load the selected pre-trained model and cache it to avoid repeated file loading.
@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[name]))

# Generate predictions and calculate six performance metrics; cached per model/data pair.
@st.cache_data
def compute_metrics(model_name, X, y):
    model = load_model(model_name)
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
st.title("\U0001F6D2 Online Shoppers Purchase Intention - Model Evaluator")
st.caption(
    "Uses the bundled `test_data.csv` by default, or upload your own **test CSV** "
    "containing the 17 session features and the `Revenue` column (1/True = purchase, "
    "0/False = no purchase) in the sidebar. The saved models are loaded and evaluated "
    "on it - nothing is retrained."
)


# Path to the bundled sample dataset shipped alongside the app.
BUNDLED_DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data.csv")

# Provide controls for uploading the test CSV and selecting a pre-trained model.
with st.sidebar:
    st.header("\u2699\uFE0F Controls")
    st.subheader("Test data")
    uploaded = st.file_uploader("Upload test CSV", type=["csv"])
    use_bundled = st.checkbox("Use bundled test_data.csv", value=True)
    if os.path.exists(BUNDLED_DATA_PATH):
        with open(BUNDLED_DATA_PATH, "rb") as _sample_file:
            st.download_button(
                "\u2B07\uFE0F Download sample CSV",
                _sample_file.read(),
                file_name="test_data.csv",
                mime="text/csv",
            )
    model_name = st.selectbox("Select model", list(MODEL_FILES.keys()))
    with st.expander("Required columns"):
        st.caption(", ".join(FEATURES + [TARGET]))

# An uploaded file always takes priority; otherwise fall back to the bundled sample.
if uploaded is not None:
    raw = pd.read_csv(uploaded)
    source_label = uploaded.name
elif use_bundled and os.path.exists(BUNDLED_DATA_PATH):
    raw = pd.read_csv(BUNDLED_DATA_PATH)
    source_label = "test_data.csv (bundled)"
else:
    st.info(
        "\U0001F446 Upload a CSV in the sidebar, or check 'Use bundled test_data.csv' to begin."
    )
    st.stop()

# Validate the presence of all required feature and target columns.
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

# Display a collapsible preview of the dataset in use and its dimensions.
with st.expander(f"\U0001F4C4 Data preview - {source_label} ({X.shape[0]} rows, {X.shape[1]} features)"):
    st.dataframe(raw.head(10), width="stretch")

st.divider()

# Evaluate the selected model's performance on the uploaded test dataset.
pred, metrics = compute_metrics(model_name, X, y)

# Display the six evaluation metrics of the selected machine learning model.
st.subheader(f"\U0001F4CA Evaluation metrics - {model_name}")
with st.container(border=True):
    for col, (k, v) in zip(st.columns(6), metrics.items()):
        col.metric(k, f"{v:.4f}")


# Divide the interface into two sections for displaying evaluation visualizations.
left, right = st.columns(2)
with left:
    with st.container(border=True):
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
        plt.close(fig)  # avoid leaking figures across reruns
with right:
    with st.container(border=True):
        # Generate and display the classification report containing precision, recall, F1-score, and support.
        st.subheader("Classification report")
        rep = classification_report(
            y, pred, target_names=LABELS, output_dict=True, zero_division=0
        )
        st.dataframe(pd.DataFrame(rep).transpose().round(4), width="stretch")

st.divider()

# Evaluate all five pre-trained models on the same test dataset for performance comparison.
st.subheader("\U0001F3C6 All models on the uploaded data")
rows = []
for name in MODEL_FILES:
    _, m = compute_metrics(name, X, y)
    rows.append({"ML Model Name": name, **m})
# Sort by F1 so the best-performing model surfaces first, and highlight it.
comparison = pd.DataFrame(rows).round(4).sort_values("F1", ascending=False, ignore_index=True)
best_model = comparison.loc[0, "ML Model Name"]
st.caption(f"Best performing model by F1 score: **{best_model}**")
metric_cols = [c for c in comparison.columns if c != "ML Model Name"]


# Bold the overall best row (row 0, after sorting by F1) in addition to per-metric green leaders.
def _highlight_best_row(row):
    style = "font-weight: 700" if row.name == 0 else ""
    return [style] * len(row)


st.dataframe(
    comparison.style
    .apply(_highlight_best_row, axis=1)
    .highlight_max(subset=metric_cols, color="#c6f6d5"),
    width="stretch",
    hide_index=True,
)
st.download_button(
    "\u2B07\uFE0F Download comparison as CSV",
    comparison.to_csv(index=False).encode("utf-8"),
    file_name="model_comparison.csv",
    mime="text/csv",
)
