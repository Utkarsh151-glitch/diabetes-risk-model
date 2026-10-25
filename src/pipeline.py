"""Feature definitions and the preprocessing + model pipeline shared by training, the API and the tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "pima_diabetes.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET = "Outcome"
# In this dataset a 0 in these columns means "not measured", not a real value.
ZERO_MEANS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
SEED = 42


class ZeroToNaN(BaseEstimator, TransformerMixin):
    """Replace 0 with NaN in the given columns so the imputer treats them as missing."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=FEATURES).copy()
        cols = self.columns if self.columns is not None else ZERO_MEANS_MISSING
        X[cols] = X[cols].replace(0, np.nan)
        return X


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"data is missing columns: {sorted(missing)}")
    return df[FEATURES + [TARGET]]


def build_pipeline(model, add_indicator: bool = False) -> Pipeline:
    """Zero-as-missing -> median imputation (refit inside every CV fold) -> scaling -> model."""
    return Pipeline([
        ("zeros", ZeroToNaN(ZERO_MEANS_MISSING)),
        ("impute", SimpleImputer(strategy="median", add_indicator=add_indicator)),
        ("scale", StandardScaler()),
        ("model", model),
    ])
