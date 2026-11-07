"""
ML Assignment 2 - Model training and evaluation
Dataset: UCI Online Shoppers Purchasing Intention Dataset

"""

import os

import joblib
import pandas as pd
from data_utils import CATEGORICAL, NUMERIC, TARGET, prepare
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Set a fixed random seed to ensure reproducible training and testing results.
SEED = 42
# Load the original dataset and preprocess it into input features and target values.
raw = pd.read_csv("data/online_shoppers_intention.csv")
X, y = prepare(raw)

# Split the dataset into 80% training and 20% testing while maintaining
# the purchase/no-purchase class distribution through stratification.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)

# Save the test dataset with its target column for independent evaluation
# through the Streamlit application.
test_df = X_test.copy()
test_df[TARGET] = y_test.values
test_df.to_csv("test_data.csv", index=False)

# Create a preprocessing pipeline to standardize numerical features
# and convert categorical features into numerical representations.
def make_preprocessor():
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
        ]
    )


# Define the five machine learning classification algorithms required.
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=11),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=SEED),
}

# Create a directory to store the trained machine learning models.
os.makedirs("model", exist_ok=True)
rows = []

# Train and evaluate each classifier using the same training and testing datasets.
for name, clf in models.items():

    # Combine preprocessing and the classifier into a single pipeline
    # to ensure consistent data transformation during training and prediction.
    pipe = Pipeline([("prep", make_preprocessor()), ("clf", clf)])

    # Train the model and generate predictions and purchase probabilities on test data.
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    # Calculate the six required performance metrics for each classifier.
    rows.append(
        {
            "ML Model Name": name,
            "Accuracy": accuracy_score(y_test, pred),
            "AUC": roc_auc_score(y_test, proba),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "MCC": matthews_corrcoef(y_test, pred),
        }
    )

    # Save the complete trained pipeline, including preprocessing and classifier,
    # for future use in the Streamlit application.
    joblib.dump(pipe, os.path.join("model", name.lower().replace(" ", "_") + ".joblib"))

# Combine all model evaluation results into a comparison table
# and save the metrics for further analysis.
res = pd.DataFrame(rows).round(4)
res.to_csv("model/metrics.csv", index=False)

# Display dataset dimensions, training and testing sizes, class distribution,
# and the final performance comparison of all five classifiers.
print(
    f"Rows: {len(X)} | train: {len(X_train)} | test: {len(X_test)} | features: {X.shape[1]}"
)
print(f"Positive rate train {y_train.mean():.4f} test {y_test.mean():.4f}")
print(res.to_string(index=False))
