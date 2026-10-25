"""Train and evaluate the diabetes risk model on the Pima Indians Diabetes Database.

Protocol (seed 42):
1. Load data, drop exact duplicates.
2. Stratified 80/20 split FIRST. The test split is used once, at the very end.
3. Every candidate is a Pipeline (zero-as-missing -> median imputer -> scaler -> model), so the imputer and
   scaler are refit inside every CV fold.
4. Model selection: RepeatedStratifiedKFold(5 splits x 5 repeats) ROC-AUC on the training split only.
   If the best model does not beat logistic regression by more than one standard deviation, keep logistic
   regression (simpler, interpretable).
5. Decision threshold: maximise Youden's J (TPR - FPR) on out-of-fold training predictions.
6. Fit the chosen pipeline on the full training split, evaluate once on the test split, save it.

usage: python src/train.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import (  # noqa: E402
    DATA_PATH,
    FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    ROOT,
    SEED,
    TARGET,
    ZERO_MEANS_MISSING,
    build_pipeline,
    load_data,
)

N_SPLITS, N_REPEATS = 5, 5
N_BOOTSTRAP = 1000
DOCS = ROOT / "docs"


def candidates() -> list[dict]:
    """Every configuration evaluated in CV. `family` groups them for the comparison table."""
    import xgboost as xgb

    out = []
    for ind in (False, True):
        for c in (0.1, 1.0):
            out.append({"family": "LogisticRegression", "name": f"LogReg C={c}", "indicator": ind,
                        "params": {"C": c},
                        "make": lambda c=c: LogisticRegression(C=c, max_iter=2000)})
        for depth in (None, 6):
            for leaf in (1, 5):
                out.append({"family": "RandomForest", "name": f"RF depth={depth} leaf={leaf}", "indicator": ind,
                            "params": {"n_estimators": 500, "max_depth": depth, "min_samples_leaf": leaf},
                            "make": lambda d=depth, l=leaf: RandomForestClassifier(
                                n_estimators=500, max_depth=d, min_samples_leaf=l, random_state=SEED, n_jobs=-1)})
        for depth in (2, 3):
            for lr in (0.05, 0.1):
                out.append({"family": "XGBoost", "name": f"XGB depth={depth} lr={lr}", "indicator": ind,
                            "params": {"n_estimators": 200, "max_depth": depth, "learning_rate": lr,
                                       "subsample": 0.8, "colsample_bytree": 0.8},
                            "make": lambda d=depth, r=lr: xgb.XGBClassifier(
                                n_estimators=200, max_depth=d, learning_rate=r, subsample=0.8,
                                colsample_bytree=0.8, eval_metric="logloss", random_state=SEED, n_jobs=4)})
    for cand in out:
        cand["name"] += " +indicators" if cand["indicator"] else ""
    return out


def split(df: pd.DataFrame):
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def cv_scores_and_oof(pipe, X: pd.DataFrame, y: pd.Series):
    """ROC-AUC per fold and the out-of-fold probability for every training row (mean over repeats)."""
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    scores, oof_sum, oof_n = [], np.zeros(len(y)), np.zeros(len(y))
    for tr, va in cv.split(X, y):
        model = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
        p = model.predict_proba(X.iloc[va])[:, 1]
        scores.append(roc_auc_score(y.iloc[va], p))
        oof_sum[va] += p
        oof_n[va] += 1
    return np.array(scores), oof_sum / oof_n


def youden_threshold(y_true, prob) -> float:
    fpr, tpr, thr = roc_curve(y_true, prob)
    j = tpr - fpr
    best = int(np.argmax(j))
    return float(thr[best])


def bootstrap_auc_ci(y_true, prob, n=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    aucs = []
    while len(aucs) < n:
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue
        aucs.append(roc_auc_score(y_true[idx], prob[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def select(cv_rows: list[dict]) -> tuple[dict, str]:
    best_per_family = {}
    for row in cv_rows:
        f = row["family"]
        if f not in best_per_family or row["cv_auc_mean"] > best_per_family[f]["cv_auc_mean"]:
            best_per_family[f] = row
    lr = best_per_family["LogisticRegression"]
    best = max(best_per_family.values(), key=lambda r: r["cv_auc_mean"])
    if best is lr:
        return lr, "Logistic regression has the highest mean CV ROC-AUC."
    margin = best["cv_auc_mean"] - lr["cv_auc_mean"]
    if margin <= best["cv_auc_std"]:
        return lr, (f"{best['name']} beats logistic regression by {margin:.4f} mean CV ROC-AUC, which is not more "
                    f"than one standard deviation ({best['cv_auc_std']:.4f}), so the simpler, interpretable "
                    f"logistic regression is kept.")
    return best, (f"{best['name']} beats logistic regression by {margin:.4f} mean CV ROC-AUC, more than one "
                  f"standard deviation ({best['cv_auc_std']:.4f}).")


def fit_final(cand: dict, X_tr, y_tr, X_te, y_te) -> tuple:
    """Threshold from OOF predictions, then one fit on the training split and one test evaluation."""
    pipe = build_pipeline(cand["make"](), add_indicator=cand["indicator"])
    scores, oof = cv_scores_and_oof(pipe, X_tr, y_tr)
    threshold = youden_threshold(y_tr, oof)
    model = clone(pipe).fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    pred = (p >= threshold).astype(int)
    lo, hi = bootstrap_auc_ci(y_te, p)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    test = {
        "roc_auc": round(float(roc_auc_score(y_te, p)), 4),
        "roc_auc_ci95": [round(lo, 4), round(hi, 4)],
        "pr_auc": round(float(average_precision_score(y_te, p)), 4),
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "precision": round(float(precision_score(y_te, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_te, pred)), 4),
        "f1": round(float(f1_score(y_te, pred)), 4),
        "brier": round(float(brier_score_loss(y_te, p)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "positive_rate": round(float(np.mean(y_te)), 4),
    }
    oof_pred = (oof >= threshold).astype(int)
    oof_stats = {
        "sensitivity": round(float(recall_score(y_tr, oof_pred)), 4),
        "specificity": round(float(recall_score(1 - y_tr, 1 - oof_pred)), 4),
    }
    return model, threshold, test, oof_stats, p, scores


def plots(model, cand, X_te, y_te, p, threshold):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    DOCS.mkdir(exist_ok=True)
    auc = roc_auc_score(y_te, p)
    fpr, tpr, _ = roc_curve(y_te, p)
    tn, fp, fn, tp = confusion_matrix(y_te, (p >= threshold).astype(int), labels=[0, 1]).ravel()
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot(fpr, tpr, lw=2, color="#1f5fa8", label=f"Test ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#999", label="Chance")
    ax.scatter([fp / (fp + tn)], [tp / (tp + fn)], color="#c0392b", zorder=3, label=f"Threshold {threshold:.3f}")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (sensitivity)",
           title="ROC curve on the held-out test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(DOCS / "roc_curve.png", dpi=150)
    plt.close(fig)

    frac_pos, mean_pred = calibration_curve(y_te, p, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="#999", label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, marker="o", lw=2, color="#1f5fa8", label="Model (8 quantile bins)")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction with diabetes",
           title="Calibration on the held-out test set", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(DOCS / "calibration.png", dpi=150)
    plt.close(fig)

    est = model.named_steps["model"]
    if isinstance(est, LogisticRegression):
        names = list(model.named_steps["impute"].get_feature_names_out(FEATURES))
        vals = est.coef_[0]
        title, xlabel = "Logistic regression coefficients", "Coefficient (per 1 SD of the feature)"
        importance = dict(zip(names, map(float, vals)))
    else:
        r = permutation_importance(model, X_te, y_te, scoring="roc_auc", n_repeats=30, random_state=SEED)
        names, vals = FEATURES, r.importances_mean
        title, xlabel = "Permutation importance (test set)", "Drop in ROC-AUC when shuffled"
        importance = dict(zip(names, map(float, vals)))
    order = np.argsort(np.abs(vals))
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(names) + 1.4))
    ax.barh([names[i] for i in order], [vals[i] for i in order],
            color=["#c0392b" if vals[i] > 0 else "#1f5fa8" for i in order])
    ax.axvline(0, color="#555", lw=0.8)
    ax.set(title=title, xlabel=xlabel)
    fig.tight_layout()
    fig.savefig(DOCS / "feature_importance.png", dpi=150)
    plt.close(fig)
    return {k: round(v, 4) for k, v in importance.items()}


def versions() -> dict:
    import xgboost

    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__}


def main():
    raw = load_data()
    df = raw.drop_duplicates()
    dropped = len(raw) - len(df)
    print(f"rows {len(raw)}, exact duplicates dropped: {dropped}")
    X_tr, X_te, y_tr, y_te = split(df)
    print(f"train {len(y_tr)} ({int(y_tr.sum())} positive), test {len(y_te)} ({int(y_te.sum())} positive)")

    cands = candidates()
    cv_rows = []
    for cand in cands:
        pipe = build_pipeline(cand["make"](), add_indicator=cand["indicator"])
        scores, _ = cv_scores_and_oof(pipe, X_tr, y_tr)
        row = {"family": cand["family"], "name": cand["name"], "indicator": cand["indicator"],
               "params": cand["params"], "cv_auc_mean": round(float(scores.mean()), 4),
               "cv_auc_std": round(float(scores.std(ddof=1)), 4)}
        cv_rows.append(row)
        print(f"  {row['name']:<44} CV ROC-AUC {row['cv_auc_mean']:.4f} ± {row['cv_auc_std']:.4f}")

    chosen_row, reason = select(cv_rows)
    cand = next(c for c in cands if c["name"] == chosen_row["name"])
    print(f"chosen: {cand['name']}. {reason}")

    model, threshold, test, oof_stats, p, _ = fit_final(cand, X_tr, y_tr, X_te, y_te)
    print(f"threshold {threshold:.4f} | test {json.dumps(test)}")
    importance = plots(model, cand, X_te, y_te, p, threshold)

    medians = dict(zip(FEATURES, map(float, model.named_steps["impute"].statistics_)))
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    metrics = {
        "model": {"family": cand["family"], "name": cand["name"], "params": cand["params"],
                  "missing_indicators": cand["indicator"], "selection_reason": reason},
        "data": {"file": "data/pima_diabetes.csv",
                 "sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
                 "rows": len(raw), "exact_duplicates_dropped": dropped,
                 "train_rows": len(y_tr), "train_positive": int(y_tr.sum()),
                 "test_rows": len(y_te), "test_positive": int(y_te.sum()),
                 "zero_means_missing": ZERO_MEANS_MISSING,
                 "training_medians_after_zero_to_missing": {k: round(v, 4) for k, v in medians.items()}},
        "protocol": {"seed": SEED, "split": "stratified 80/20, test used once",
                     "cv": f"RepeatedStratifiedKFold(n_splits={N_SPLITS}, n_repeats={N_REPEATS}) ROC-AUC on the training split"},
        "cv": cv_rows,
        "threshold": {"value": round(threshold, 4),
                      "rule": "Maximises Youden's J (sensitivity + specificity - 1) on out-of-fold predictions "
                              "for the training split (mean over the 5 repeats); the test set is not used.",
                      "out_of_fold": oof_stats},
        "test": {**test, "bootstrap_resamples": N_BOOTSTRAP},
        "feature_importance": importance,
        "versions": versions(),
        "trained_on": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"saved {MODEL_PATH.relative_to(ROOT)} and {METRICS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
