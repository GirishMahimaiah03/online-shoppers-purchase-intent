"""Shared feature definitions and cleaning used by training script and Streamlit app."""
import pandas as pd

# Define the target column and separate numerical and categorical input features.
TARGET = "Revenue"
NUMERIC = ["Administrative", "Administrative_Duration", "Informational",
           "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
           "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Weekend"]

# Treat Month and coded ID columns as categorical variables because
# their numerical values represent categories rather than measurable quantities.
CATEGORICAL = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType"]

# Combine numerical and categorical columns to define all 17 input features.
FEATURES = NUMERIC + CATEGORICAL

# Convert Boolean and binary values into integer format (1 = True, 0 = False).
def _to_binary(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"]).astype(int)

# Preprocess the dataset by selecting required columns, converting data types,
# and separating input features (X) from the purchase target (y).
def prepare(df):
    """Return (X, y) from a dataframe holding all 17 features and Revenue."""
     # Select the required feature and target columns without modifying the original dataset.
    out = df[FEATURES + [TARGET]].copy()
     # Convert the Weekend feature and Revenue target into binary integer values.
    out["Weekend"] = _to_binary(out["Weekend"])
    out[TARGET] = _to_binary(out[TARGET])

    # Convert coded categorical variables into string values so that
    # machine learning preprocessing treats them as categories.
    for c in ["OperatingSystems", "Browser", "Region", "TrafficType"]:
        out[c] = pd.to_numeric(out[c]).astype(int).astype(str)

    # Return the processed input features and target variable separately.
    return out[FEATURES], out[TARGET]
