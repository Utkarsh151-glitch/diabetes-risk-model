import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.pipeline import FEATURES, METRICS_PATH, MODEL_PATH, ZERO_MEANS_MISSING, ZeroToNaN, build_pipeline, load_data
from src.train import candidates, fit_final, split


@pytest.fixture(scope="module")
def data():
    df = load_data()
    return df, split(df)


def test_dataset_is_the_expected_pima_file(data):
    df, _ = data
    assert df.shape == (768, 9)
    assert df["Outcome"].value_counts().to_dict() == {0: 500, 1: 268}


def test_train_and_test_do_not_overlap(data):
    _, (X_tr, X_te, y_tr, y_te) = data
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert len(X_tr) + len(X_te) == 768
    assert len(X_te) == 154
    # stratified: both splits keep the overall positive rate (268/768)
    assert abs(y_tr.mean() - y_te.mean()) < 0.01


def test_zeros_become_nan_only_in_intended_columns():
    zeros = pd.DataFrame([[0.0] * len(FEATURES)], columns=FEATURES)
    out = ZeroToNaN(ZERO_MEANS_MISSING).transform(zeros)
    for col in FEATURES:
        if col in ZERO_MEANS_MISSING:
            assert np.isnan(out[col].iloc[0]), col
        else:
            assert out[col].iloc[0] == 0, col


def test_imputer_medians_come_from_training_data_only(data):
    _, (X_tr, X_te, y_tr, _) = data
    pipe = build_pipeline(LogisticRegression(max_iter=2000)).fit(X_tr, y_tr)
    train_medians = X_tr[FEATURES].replace({c: {0: np.nan} for c in ZERO_MEANS_MISSING}).median()
    np.testing.assert_allclose(pipe.named_steps["impute"].statistics_, train_medians.to_numpy())
    # the saved model was fit on the same training rows
    saved = joblib.load(MODEL_PATH)
    np.testing.assert_allclose(saved.named_steps["impute"].statistics_, train_medians.to_numpy())


def test_retraining_reproduces_saved_metrics(data):
    _, (X_tr, X_te, y_tr, y_te) = data
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    cand = next(c for c in candidates() if c["name"] == metrics["model"]["name"])
    _, threshold, test, _, _, cv_scores = fit_final(cand, X_tr, y_tr, X_te, y_te)
    tol = 1e-3
    assert abs(threshold - metrics["threshold"]["value"]) < tol
    chosen_cv = next(r for r in metrics["cv"] if r["name"] == cand["name"])
    assert abs(cv_scores.mean() - chosen_cv["cv_auc_mean"]) < tol
    for key in ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "brier"):
        assert abs(test[key] - metrics["test"][key]) < tol, key
    assert test["confusion_matrix"] == metrics["test"]["confusion_matrix"]
