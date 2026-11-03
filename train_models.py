
from pathlib import Path
import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

RANDOM_STATE = 42
N_SPLITS = 5
TEST_FOLD = 0

NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
]

BOOLEAN_FEATURES = ["Weekend"]
TARGET_COLUMN = "Revenue"

EXPECTED_COLUMNS = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
    + BOOLEAN_FEATURES
    + [TARGET_COLUMN]
)


def load_and_validate_data(csv_path: Path):
    """Load the CSV and validate the dataset schema and target."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if df.shape != (12330, 18):
        warnings.warn(
            f"Expected the verified UCI file shape (12330, 18), "
            f"but received {df.shape}. Check that the correct CSV was supplied."
        )

    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    extra_columns = sorted(set(df.columns) - set(EXPECTED_COLUMNS))

    if missing_columns or extra_columns:
        raise ValueError(
            "Dataset columns do not match the verified schema. "
            f"Missing columns: {missing_columns}; extra columns: {extra_columns}"
        )

    # Keep the verified source column order explicit.
    df = df[EXPECTED_COLUMNS].copy()

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("The target column Revenue contains missing values.")

    target_values = set(df[TARGET_COLUMN].unique())
    if not target_values.issubset({True, False, 0, 1}):
        raise ValueError(
            f"Unexpected Revenue values: {sorted(map(str, target_values))}"
        )

    if df.isna().any().any():
        missing_counts = df.isna().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(
            "Missing values were found. Do not silently impute or delete them. "
            f"Column counts: {missing_counts}"
        )

    # Explicitly encode the Boolean target as 0 = no purchase, 1 = purchase.
    y = df[TARGET_COLUMN].astype(int)
    X = df.drop(columns=[TARGET_COLUMN]).copy()

    return df, X, y


def make_preprocessor():
    """Create preprocessing that is fitted only on training data."""
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
            ("boolean", "passthrough", BOOLEAN_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0,
        verbose_feature_names_out=True,
    )

    return preprocessor


def make_models():
    """Return the five classifiers required by the assignment."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
        ),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(
            probability=True,
            random_state=RANDOM_STATE,
        ),
    }


def calculate_metrics(y_true, y_pred, y_probability):
    """Calculate all six metrics required by the assignment."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_probability),
        "Precision": precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "F1": f1_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate five classifiers for ML Assignment 2."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("online_shoppers_intention.csv"),
        help="Path to online_shoppers_intention.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("."),
        help="Project output directory (default: current directory)",
    )
    args = parser.parse_args()

    output_dir = args.output.resolve()
    model_dir = output_dir / "model"
    results_dir = output_dir / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and validate the original dataset.
    df, X, y = load_and_validate_data(args.data)

    print("\nDATASET VALIDATION")
    print("------------------")
    print(f"Dataset shape: {df.shape}")
    print(f"Input feature count: {X.shape[1]}")
    print(f"Exact duplicate rows retained: {int(df.duplicated().sum())}")
    print("Target distribution:")
    print(y.value_counts().sort_index().rename(index={0: "No purchase", 1: "Purchase"}))

    # 2. Build groups from identical input-feature rows.
    # Rows with identical X values must remain in the same split.
    groups = pd.factorize(pd.MultiIndex.from_frame(X), sort=False)[0]

    # Verify that an identical feature group does not have conflicting labels.
    group_target_counts = (
        pd.DataFrame({"group": groups, "target": y.to_numpy()})
        .groupby("group")["target"]
        .nunique()
    )
    if (group_target_counts > 1).any():
        raise ValueError(
            "At least one identical feature group has conflicting target labels. "
            "Review these records before splitting."
        )

    # 3. Create one stratified, group-aware 80/20 split.
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    splits = list(splitter.split(X, y, groups=groups))
    train_indices, test_indices = splits[TEST_FOLD]

    X_train = X.iloc[train_indices].copy()
    X_test = X.iloc[test_indices].copy()
    y_train = y.iloc[train_indices].copy()
    y_test = y.iloc[test_indices].copy()

    train_groups = set(groups[train_indices])
    test_groups = set(groups[test_indices])
    overlap = train_groups.intersection(test_groups)

    if overlap:
        raise RuntimeError("Duplicate feature groups crossed the train/test split.")

    print("\nTRAIN/TEST SPLIT")
    print("----------------")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Training target counts: {y_train.value_counts().sort_index().to_dict()}")
    print(f"Testing target counts: {y_test.value_counts().sort_index().to_dict()}")
    print(f"Overlapping duplicate groups: {len(overlap)}")

    # 4. Save the held-out test data with the target for Streamlit evaluation.
    test_data = X_test.copy()
    test_data[TARGET_COLUMN] = y_test.to_numpy()
    test_data.to_csv(output_dir / "test_data.csv", index=False)

    # 5. Train each model using the same training and test partitions.
    metric_rows = []
    fitted_models = make_models()

    print("\nMODEL TRAINING AND EVALUATION")
    print("-----------------------------")

    for model_name, classifier in fitted_models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                ("classifier", classifier),
            ]
        )

        pipeline.fit(X_train, y_train)

        # Predicted labels are used for threshold-based classification metrics.
        y_pred = pipeline.predict(X_test)

        # All five selected classifiers support predict_proba in this setup.
        # Positive-class probability is used for binary ROC-AUC.
        positive_class_index = list(pipeline.classes_).index(1)
        y_probability = pipeline.predict_proba(X_test)[:, positive_class_index]

        metrics = calculate_metrics(y_test, y_pred, y_probability)
        metric_rows.append({"Model": model_name, **metrics})

        # Save the entire pipeline, including fitted preprocessing.
        safe_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )
        model_path = model_dir / f"{safe_name}.joblib"
        joblib.dump(pipeline, model_path)

        # Save confusion matrix and classification report for inspection.
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        pd.DataFrame(
            cm,
            index=["Actual_No_Purchase", "Actual_Purchase"],
            columns=["Predicted_No_Purchase", "Predicted_Purchase"],
        ).to_csv(results_dir / f"{safe_name}_confusion_matrix.csv")

        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["No purchase", "Purchase"],
            zero_division=0,
            output_dict=True,
        )
        with open(
            results_dir / f"{safe_name}_classification_report.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(report, file, indent=4)

        print(f"\n{model_name}")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name:10s}: {metric_value:.6f}")
        print(f"Saved model: {model_path}")

    # 6. Save the common comparison table.
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(results_dir / "comparison_metrics.csv", index=False)

    # 7. Record split and dataset details for reproducibility.
    metadata = {
        "dataset": "UCI Online Shoppers Purchasing Intention Dataset",
        "source_file": str(args.data),
        "original_records": int(len(df)),
        "input_features": int(X.shape[1]),
        "exact_duplicate_rows_retained": int(df.duplicated().sum()),
        "unique_feature_groups": int(len(np.unique(groups))),
        "splitter": "StratifiedGroupKFold",
        "n_splits": N_SPLITS,
        "selected_test_fold": TEST_FOLD,
        "random_state": RANDOM_STATE,
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "training_target_counts": {
            str(k): int(v) for k, v in y_train.value_counts().to_dict().items()
        },
        "testing_target_counts": {
            str(k): int(v) for k, v in y_test.value_counts().to_dict().items()
        },
        "overlapping_duplicate_groups": int(len(overlap)),
        "positive_class_for_metrics": 1,
        "auc_method": "ROC-AUC using predicted probability for Revenue=True",
    }

    with open(output_dir / "split_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\nFINAL MODEL COMPARISON")
    print("----------------------")
    print(metrics_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("\nOUTPUTS CREATED")
    print("---------------")
    print(f"Saved models: {model_dir}")
    print(f"Metrics and reports: {results_dir}")
    print(f"Test data: {output_dir / 'test_data.csv'}")
    print(f"Split metadata: {output_dir / 'split_metadata.json'}")


if __name__ == "__main__":
    main()
