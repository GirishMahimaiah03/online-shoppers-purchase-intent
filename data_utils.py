"""Shared feature definitions and cleaning used by training script and Streamlit app."""
import pandas as pd

TARGET = "Revenue"
NUMERIC = ["Administrative", "Administrative_Duration", "Informational",
           "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
           "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Weekend"]
# Month and the four coded IDs are treated as categories (their numeric order has no meaning)
CATEGORICAL = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType"]
FEATURES = NUMERIC + CATEGORICAL


def _to_binary(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"]).astype(int)


def prepare(df):
    """Return (X, y) from a dataframe holding all 17 features and Revenue."""
    out = df[FEATURES + [TARGET]].copy()
    out["Weekend"] = _to_binary(out["Weekend"])
    out[TARGET] = _to_binary(out[TARGET])
    for c in ["OperatingSystems", "Browser", "Region", "TrafficType"]:
        out[c] = pd.to_numeric(out[c]).astype(int).astype(str)
    return out[FEATURES], out[TARGET]
