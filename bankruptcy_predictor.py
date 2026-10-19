"""Bankruptcy prediction for Polish companies, benchmarked against Altman Z'.

Run:  python bankruptcy_predictor.py
Needs: pandas, numpy, scikit-learn, matplotlib
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_PATH = "csv_result-2year.csv"   # <- change this if your CSV lives elsewhere
SEED = 42

# Altman Z' inputs (1983 private-firm version; no market value of equity needed)
Z_COLS = ["Attr3", "Attr6", "Attr7", "Attr8", "Attr9"]


# ---------------------------------------------------------------- data
def load_data(path: str) -> pd.DataFrame:
    """Load CSV, treat '?' as missing, force numeric, drop id/duplicates/Attr36."""
    df = pd.read_csv(path, na_values=["?"], low_memory=False)
    feats = [c for c in df.columns if c.startswith("Attr")]
    df[feats] = df[feats].apply(pd.to_numeric, errors="coerce")
    df = df.drop(columns=["id", "Attr36"], errors="ignore")  # Attr36 ~= Attr9
    df = df.drop_duplicates().reset_index(drop=True)          # dedupe BEFORE splitting
    return df


def altman_z_prime(df: pd.DataFrame) -> pd.Series:
    """Z' = 0.717*X1 + 0.847*X2 + 3.107*X3 + 0.420*X4 + 0.998*X5."""
    return (0.717 * df["Attr3"]      # X1 working capital / total assets
            + 0.847 * df["Attr6"]    # X2 retained earnings / total assets
            + 3.107 * df["Attr7"]    # X3 EBIT / total assets
            + 0.420 * df["Attr8"]    # X4 book equity / total liabilities
            + 0.998 * df["Attr9"])   # X5 sales / total assets


def z_zone(z: pd.Series) -> pd.Series:
    """Standard Z' zones: <1.23 distress, 1.23-2.90 grey, >2.90 safe."""
    return pd.cut(z, [-np.inf, 1.23, 2.90, np.inf],
                  labels=["distress (<1.23)", "grey (1.23-2.90)", "safe (>2.90)"])


# ---------------------------------------------------------------- model parts
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to its 1st/99th percentile, learned on training data only."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanpercentile(X, 1, axis=0)
        self.hi_ = np.nanpercentile(X, 99, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def linear_prep() -> list:
    """Clip outliers -> median impute (+ missing flags) -> scale."""
    return [("clip", Winsorizer()),
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler())]


def build_models(feature_cols: list) -> dict:
    """The three trainable models. Every preprocessing step lives inside the Pipeline."""
    lr = LogisticRegression(class_weight="balanced", C=0.1, max_iter=2000)
    z_only = ColumnTransformer([("z", Pipeline(linear_prep()), Z_COLS)])
    return {
        "Logistic (5 Z' components)": Pipeline([("prep", z_only), ("clf", clone(lr))]),
        "Logistic (all features)": Pipeline(linear_prep() + [("clf", clone(lr))]),
        "HistGradientBoosting": RandomizedSearchCV(
            HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
            param_distributions={
                "learning_rate": [0.02, 0.05, 0.1],
                "max_depth": [3, 4, 6, None],
                "max_leaf_nodes": [8, 15, 31],
                "min_samples_leaf": [10, 20, 40],
                "l2_regularization": [0.0, 0.1, 1.0, 10.0],
                "max_iter": [200, 400],
            },
            n_iter=20, scoring="average_precision", random_state=SEED,
            cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1),
    }


# ---------------------------------------------------------------- reusable helpers
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn any frame with the raw Attr columns into model input (adds Z' if missing)."""
    X = df.drop(columns=["class", "id", "Attr36"], errors="ignore").copy()
    X = X.apply(pd.to_numeric, errors="coerce")        # '?' strings -> NaN
    if "Z_prime" not in X.columns:
        X["Z_prime"] = altman_z_prime(X)               # needs Attr3, 6, 7, 8, 9
    return X


def best_f2_threshold(y_true, proba) -> float:
    """Probability cut-off that maximizes F2 (recall counts 4x more than precision)."""
    p, r, t = precision_recall_curve(y_true, proba)
    f2 = 5 * p[:-1] * r[:-1] / np.maximum(4 * p[:-1] + r[:-1], 1e-12)
    return t[np.argmax(f2)]


def fit_final_model(train_df: pd.DataFrame, y_train):
    """Fit the gradient-boosting + Z' model on ANY training frame. Returns (model, threshold)."""
    X = prepare_features(train_df)
    search = build_models(list(X.columns))["HistGradientBoosting"]
    search.fit(X, y_train)
    est = search.best_estimator_
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(clone(est), X, y_train, cv=cv, method="predict_proba")[:, 1]
    return est, best_f2_threshold(y_train, oof)


def predict_labels(est, thr: float, df: pd.DataFrame) -> np.ndarray:
    """0/1 bankruptcy predictions for any frame with the raw Attr columns."""
    return (est.predict_proba(prepare_features(df))[:, 1] >= thr).astype(int)


# ---------------------------------------------------------------- main
def main():
    df = load_data(DATA_PATH)
    df["Z_prime"] = altman_z_prime(df)
    y = df.pop("class")
    print(f"Rows after dedupe: {len(df)} | bankrupt: {int(y.sum())} ({y.mean():.2%})")

    # sanity check: raw Z' alone should give ROC-AUC ~0.61 on this file
    z_all = df["Z_prime"].fillna(df["Z_prime"].median())
    auc_all = roc_auc_score(y, -z_all)
    print(f"Sanity check, raw Z' ROC-AUC (full data): {auc_all:.3f} (expected ~0.61)")
    if abs(auc_all - 0.61) > 0.03:
        print("WARNING: differs from expected by >0.03, check your column mapping.")

    # Z' bankruptcy rate per zone
    rates = (pd.DataFrame({"zone": z_zone(df["Z_prime"]), "bankrupt": y})
             .groupby("zone", observed=True)["bankrupt"].agg(["count", "mean"]))
    rates.columns = ["companies", "bankruptcy_rate"]
    print("\nActual bankruptcy rate per Z' zone (full data):")
    print(rates.round(4).to_string())

    # one split; holdout is touched once at the end
    X_tr, X_ho, y_tr, y_ho = train_test_split(
        df, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    models = build_models(list(df.columns))
    rows = []

    # baseline: Z' as a plain score (nothing to train; score per CV fold for comparability)
    z_med = X_tr["Z_prime"].median()
    z_tr = X_tr["Z_prime"].fillna(z_med)
    cv_z = np.mean([average_precision_score(y_tr.iloc[i], -z_tr.iloc[i])
                    for _, i in cv.split(X_tr, y_tr)])
    z_ho = X_ho["Z_prime"].fillna(z_med)
    rows.append({"model": "Altman Z' (score only)", "CV PR-AUC": cv_z,
                 "Holdout PR-AUC": average_precision_score(y_ho, -z_ho),
                 "Holdout ROC-AUC": roc_auc_score(y_ho, -z_ho)})

    # train + CV the three models
    fitted = {}
    for name, model in models.items():
        if isinstance(model, RandomizedSearchCV):
            model.fit(X_tr, y_tr)
            cv_score, est = model.best_score_, model.best_estimator_
        else:
            cv_score = cross_val_score(model, X_tr, y_tr, cv=cv,
                                       scoring="average_precision").mean()
            est = model.fit(X_tr, y_tr)
        fitted[name] = (est, cv_score)
        print(f"  trained: {name} (CV PR-AUC {cv_score:.4f})")

    best_name = max(fitted, key=lambda k: fitted[k][1])
    best_est = fitted[best_name][0]

    # threshold from out-of-fold training predictions, maximizing F2 (recall-weighted)
    oof = cross_val_predict(clone(best_est), X_tr, y_tr, cv=cv,
                            method="predict_proba")[:, 1]
    p, r, t = precision_recall_curve(y_tr, oof)
    f2 = 5 * p[:-1] * r[:-1] / np.maximum(4 * p[:-1] + r[:-1], 1e-12)
    thr = t[np.argmax(f2)]

    # single holdout evaluation
    for name, (est, cv_score) in fitted.items():
        pr = est.predict_proba(X_ho)[:, 1]
        rows.append({"model": name + (" *" if name == best_name else ""),
                     "CV PR-AUC": cv_score,
                     "Holdout PR-AUC": average_precision_score(y_ho, pr),
                     "Holdout ROC-AUC": roc_auc_score(y_ho, pr)})

    print("\nResults (* = best by CV PR-AUC; HGB CV score is from its tuning search, "
          "so slightly optimistic):")
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    best_pr = best_est.predict_proba(X_ho)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_ho, (best_pr >= thr).astype(int)).ravel()
    print(f"\nBest model: {best_name} | threshold (max F2 on OOF): {thr:.3f}")
    print(f"Holdout confusion matrix -> TN {tn} | FP {fp} | FN {fn} | TP {tp}")
    print(f"Caught {tp}/{tp + fn} bankruptcies; {fp} false alarms.")

    beat = rows[1:]
    best_row = next(r_ for r_ in beat if r_["model"].startswith(best_name))
    verdict = "BEAT" if best_row["Holdout PR-AUC"] > rows[0]["Holdout PR-AUC"] else "did NOT beat"
    print(f"The best model {verdict} Z' on holdout PR-AUC.")

    # plot
    plt.figure(figsize=(6, 5))
    for label, score in [(best_name, best_pr), ("Altman Z'", -z_ho)]:
        pp, rr, _ = precision_recall_curve(y_ho, score)
        plt.plot(rr, pp, label=f"{label} (AP={average_precision_score(y_ho, score):.3f})")
    plt.axhline(y_ho.mean(), ls="--", c="grey", label="Base rate")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Holdout precision-recall"); plt.legend(); plt.tight_layout()
    plt.savefig("pr_curve.png", dpi=150)
    print("Saved pr_curve.png")

    # hand everything back so other scripts can reuse the exact same split and model
    return {"model": best_est, "threshold": thr,
            "X_train": X_tr, "y_train": y_tr, "X_holdout": X_ho, "y_holdout": y_ho}


if __name__ == "__main__":
    main()