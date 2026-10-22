"""Streamlit front-end for KNNBoost on the Polish bankruptcy datasets.

Run:  streamlit run app.py
Needs Final_model.py (your KNNBoost) in the same folder. The CSVs can sit in a folder you point the
app at, or you can upload any CSV with Attr1..Attr64 + class.
"""
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from Final_model import KNNBoost

FEATURES = [f"Attr{i}" for i in range(1, 65)]
DATA_NAMES = {i: f"csv_result-{i}year.csv" for i in range(1, 6)}
DEFAULT_KNN_COLS = ["Attr1", "Attr3", "Attr6", "Attr8", "Attr9", "Attr46"]

st.set_page_config(page_title="Bankruptcy predictor", page_icon="📉", layout="wide")


# ───────────────────────────── data ─────────────────────────────
def clean_features(df):
    """Everything to float64; '?' / text -> NaN; inf -> NaN (same as data_prep_LGBM_HGB.py)."""
    return df.apply(pd.to_numeric, errors="coerce").astype("float64").replace([np.inf, -np.inf], np.nan)


@st.cache_data(show_spinner="Loading and splitting data…")
def load_split(raw: bytes, drop_dupes: bool, test_size: float, seed: int):
    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    missing = [c for c in FEATURES + ["class"] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing[:5]}{'…' if len(missing) > 5 else ''}")
    df = clean_features(df[FEATURES + ["class"]]).dropna(subset=["class"])
    if drop_dupes:  # BEFORE the split so a copy can't land in both train and test
        df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    df["class"] = df["class"].astype(int)
    train, test = train_test_split(df, test_size=test_size, stratify=df["class"], random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def dataset_summary(raw: bytes):
    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    feats = clean_features(df[[c for c in FEATURES if c in df.columns]])
    y = pd.to_numeric(df["class"], errors="coerce") if "class" in df.columns else pd.Series(dtype=float)
    return {
        "rows": len(df), "bankrupt": int((y == 1).sum()), "healthy": int((y == 0).sum()),
        "missing_pct": float(feats.isna().mean().mean() * 100) if len(feats.columns) else 0.0,
        "dupes": int(df[[c for c in FEATURES + ["class"] if c in df.columns]].duplicated().sum()),
        "head": df.head(8),
    }


def make_weights(y, mode, pos_w):
    if mode == "None":
        return None
    if mode == "Balanced":  # identical to the data-prep file: n / (2 * class_count)
        counts = pd.Series(y).value_counts()
        return pd.Series(y).map(len(y) / (2 * counts)).to_numpy()
    return np.where(y == 1, pos_w, 1.0)


# ───────────────────────────── model ─────────────────────────────
class KNNBoostApp(KNNBoost):
    """Your KNNBoost, with the hard-coded bits exposed.

    Differences from Final_model.KNNBoost (nothing in Final_model.py is edited):
      * booster hyper-parameters are settable;
      * the KNN feature is computed on median-imputed (optionally robust-scaled) columns, because
        the data keeps NaNs and raw Euclidean distance on NaN gives garbage neighbours;
      * the threshold can be manual or learned (your commented-out recall-target code);
      * the 5-fold CV is only run when the threshold is learned (otherwise its output is unused).
    """

    def __init__(self, k, knn_columns, model, params, scale_knn=True, seed=0):
        super().__init__(k=k, knn_columns=knn_columns, model=model)
        if model == "lgbm":
            self.booster = LGBMClassifier(learning_rate=params["lr"], n_estimators=params["n_est"],
                                          num_leaves=params["leaves"], min_child_samples=params["min_leaf"],
                                          reg_lambda=params["l2"], random_state=seed, verbose=-1)
        else:
            self.booster = HistGradientBoostingClassifier(learning_rate=params["lr"], max_iter=params["n_est"],
                                                          max_leaf_nodes=params["leaves"],
                                                          min_samples_leaf=params["min_leaf"],
                                                          l2_regularization=params["l2"], random_state=seed)
        self.scale_knn = scale_knn
        self.seed = seed

    def _fit_knn_space(self):
        tr = self.X_train[:, self.knn_columns]
        med = np.nanmedian(tr, axis=0)
        self._med = np.where(np.isnan(med), 0.0, med)
        q1, q3 = np.nanpercentile(tr, [25, 75], axis=0)
        iqr = np.where(np.isnan(q3 - q1) | ((q3 - q1) == 0), 1.0, q3 - q1)
        self._iqr = iqr
        self._ktrain = self._knn_space(self.X_train)

    def _knn_space(self, X):
        Z = np.asarray(X, dtype=float)[:, self.knn_columns]
        Z = np.where(np.isnan(Z), self._med, Z)
        if self.scale_knn:
            Z = np.clip((Z - self._med) / self._iqr, -10, 10)
        return Z

    def _class_1_proportions(self, X, leave_one_out=False):
        train, X = self._ktrain, self._knn_space(X)
        k = min(self.k, len(train) - 1)
        sq_norms = np.sum(train ** 2, axis=1)
        out = np.empty(len(X))
        for start in range(0, len(X), 500):
            chunk = X[start:start + 500]
            d = np.sum(chunk ** 2, axis=1)[:, None] + sq_norms[None, :] - 2 * chunk @ train.T
            if leave_one_out:  # a training company must not be its own neighbour
                d[np.arange(len(chunk)), np.arange(start, start + len(chunk))] = np.inf
            idx = np.argpartition(d, k - 1, axis=1)[:, :k]
            out[start:start + len(chunk)] = self.y_train[idx].mean(axis=1)
        return out

    def fit(self, X, y, sample_weight=None, threshold_mode="Manual", threshold=0.10, target_recall=0.85):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y).astype(int)
        if self.knn_columns is not None:
            self._fit_knn_space()
        X_all = self._with_knn_feature(self.X_train, leave_one_out=True)
        self.threshold = float(threshold)
        if threshold_mode.startswith("Learned"):
            params = {"sample_weight": sample_weight} if sample_weight is not None else None
            cv = StratifiedKFold(5, shuffle=True, random_state=0)
            oof = cross_val_predict(clone(self.booster), X_all, self.y_train, cv=cv,
                                    method="predict_proba", params=params)[:, 1]
            _, recall, thresholds = precision_recall_curve(self.y_train, oof)
            ok = np.where(recall[:-1] >= target_recall)[0]  # strictest cutoff that still hits the recall target
            if len(ok):
                self.threshold = round(float(thresholds[ok[-1]]), 3)
        self.booster.fit(X_all, self.y_train, sample_weight=sample_weight)
        return self


def run_experiment(raw, name, cfg):
    train, test = load_split(raw, cfg["drop_dupes"], cfg["test_size"], cfg["seed"])
    feats = [f for f in FEATURES if f not in cfg["exclude"]]
    ytr = train["class"].to_numpy(int)
    knn_idx = [feats.index(c) for c in cfg["knn_cols"] if c in feats] if cfg["use_knn"] else []
    model = KNNBoostApp(cfg["k"], knn_idx or None, cfg["model"], cfg["params"], cfg["scale_knn"], cfg["seed"])
    t0 = time.time()
    model.fit(train[feats].to_numpy(float), ytr, make_weights(ytr, cfg["weight_mode"], cfg["pos_w"]),
              cfg["thr_mode"], cfg["thr"], cfg["target_recall"])
    proba = model.predict_proba(test[feats].to_numpy(float))
    return {"model": model, "feats": feats, "test": test, "proba": proba, "y": test["class"].to_numpy(int),
            "n_train": len(train), "pos_train": int(ytr.sum()), "seconds": time.time() - t0,
            "cfg": cfg, "name": name}


# ───────────────────────────── metrics ─────────────────────────────
def metrics_at(y, p, thr, beta=2.0):
    pred = p >= thr
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    b2 = beta ** 2
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "recall": rec, "precision": prec, "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "fbeta": (1 + b2) * prec * rec / (b2 * prec + rec) if b2 * prec + rec else 0.0,
        "bal_acc": balanced_accuracy_score(y, pred),
        "pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
        "prevalence": float(np.mean(y)), "flagged": int(pred.sum()),
    }


def sweep(y, p, beta):
    ts = np.linspace(0, 1, 201)
    rows = [metrics_at(y, p, t, beta) for t in ts]
    return pd.DataFrame({"threshold": ts, "Recall": [r["recall"] for r in rows],
                         "Precision": [r["precision"] for r in rows], "F1": [r["f1"] for r in rows],
                         "F-beta": [r["fbeta"] for r in rows]})


def pct(x):
    return f"{x * 100:.1f}%"


# ───────────────────────────── sidebar ─────────────────────────────
st.sidebar.title("⚙️ Controls")
st.sidebar.subheader("1 · Data")
source = st.sidebar.radio("Source", ["Local folder", "Upload CSV"], horizontal=True)
raw, data_name = None, None
if source == "Local folder":
    folder = st.sidebar.text_input("Folder containing the CSVs", value=".")
    available = [i for i in DATA_NAMES if (Path(folder) / DATA_NAMES[i]).exists()]
    if available:
        year = st.sidebar.selectbox("Dataset", available, format_func=lambda i: DATA_NAMES[i])
        raw, data_name = (Path(folder) / DATA_NAMES[year]).read_bytes(), DATA_NAMES[year]
    else:
        st.sidebar.error(f"No csv_result-<n>year.csv files found in '{folder}'.")
else:
    up = st.sidebar.file_uploader("CSV with Attr1…Attr64 and class", type="csv")
    if up is not None:
        raw, data_name = up.getvalue(), up.name

st.sidebar.subheader("2 · Model & training")
with st.sidebar.form("train_form"):
    model_name = st.selectbox("Booster", ["lgbm", "hgb"],
                              format_func=lambda m: {"lgbm": "LightGBM", "hgb": "HistGradientBoosting"}[m])
    with st.expander("Booster hyper-parameters"):
        lr = st.number_input("Learning rate", 0.005, 1.0, 0.05, 0.005, format="%.3f")
        n_est = st.number_input("Trees (n_estimators / max_iter)", 10, 3000, 300, 10)
        leaves = st.number_input("Max leaves per tree", 2, 256, 15, 1)
        min_leaf = st.number_input("Min samples per leaf", 1, 500, 20, 1)
        l2 = st.number_input("L2 regularisation", 0.0, 100.0, 1.0, 0.5)
    with st.expander("KNN feature"):
        use_knn = st.checkbox("Add KNN neighbour-fraction feature", value=False,
                              help="Off = plain booster. This is what your final_testing_file.py does, "
                                   "since KNNBoost(knn_columns=None) adds no KNN feature.")
        k = st.number_input("k (neighbours)", 1, 1000, 50, 1)
        knn_cols = st.multiselect("Columns used for distance", FEATURES, default=DEFAULT_KNN_COLS)
        scale_knn = st.checkbox("Robust-scale those columns", value=True,
                                help="Median/IQR scaling, clipped at ±10. Raw ratios have huge outliers "
                                     "that would dominate Euclidean distance.")
    with st.expander("Threshold"):
        thr_mode = st.radio("How to pick the starting threshold", ["Manual", "Learned (recall target, out-of-fold)"])
        thr_manual = st.number_input("Manual threshold", 0.0, 1.0, 0.10, 0.01)
        target_recall = st.slider("Recall target (learned mode)", 0.5, 0.99, 0.85, 0.01)
        st.caption("You can still move the threshold live after training.")
    with st.expander("Split & cleaning"):
        test_size = st.slider("Test share", 0.1, 0.5, 0.2, 0.05)
        seed = st.number_input("Random seed", 0, 10_000, 42, 1)
        drop_dupes = st.checkbox("Drop duplicate rows", value=True)
        exclude = st.multiselect("Exclude features", FEATURES, default=[])
    with st.expander("Class imbalance"):
        weight_mode = st.radio("Training sample weights", ["Balanced", "None", "Custom"])
        pos_w = st.number_input("Custom bankrupt-row weight", 1.0, 500.0, 20.0, 1.0)
    trained = st.form_submit_button("🚀 Train model", type="primary", width="stretch")

if trained:
    if raw is None:
        st.sidebar.error("Choose a dataset first.")
    elif use_knn and not [c for c in knn_cols if c not in exclude]:
        st.sidebar.error("Pick at least one (non-excluded) KNN column, or turn the KNN feature off.")
    else:
        cfg = dict(model=model_name, k=int(k), use_knn=use_knn, knn_cols=knn_cols, scale_knn=scale_knn,
                   params=dict(lr=lr, n_est=int(n_est), leaves=int(leaves), min_leaf=int(min_leaf), l2=l2),
                   thr_mode=thr_mode, thr=thr_manual, target_recall=target_recall, test_size=test_size,
                   seed=int(seed), drop_dupes=drop_dupes, exclude=exclude, weight_mode=weight_mode, pos_w=pos_w)
        try:
            with st.spinner("Training…"):
                res = run_experiment(raw, data_name, cfg)
            st.session_state["res"] = res
            st.session_state["thr_live"] = float(np.clip(res["model"].threshold, 0.0, 1.0))
            st.session_state.pop("compare", None)
        except Exception as e:  # noqa: BLE001 - show any data/model error in the UI
            st.sidebar.error(f"Training failed: {e}")


# ───────────────────────────── main ─────────────────────────────
st.title("📉 Bankruptcy prediction · KNNBoost")

if raw is not None:
    with st.expander(f"Dataset overview · {data_name}", expanded="res" not in st.session_state):
        try:
            s = dataset_summary(raw)
            c = st.columns(5)
            c[0].metric("Rows", f"{s['rows']:,}")
            c[1].metric("Bankrupt", f"{s['bankrupt']:,}")
            c[2].metric("Bankrupt share", pct(s["bankrupt"] / max(s["rows"], 1)))
            c[3].metric("Missing values", f"{s['missing_pct']:.1f}%")
            c[4].metric("Duplicate rows", f"{s['dupes']:,}")
            st.dataframe(s["head"], width="stretch")
        except Exception as e:  # noqa: BLE001
            st.error(f"Couldn't read this file: {e}")

if "res" not in st.session_state:
    st.info("Pick a dataset and press **Train model** in the sidebar.")
    st.stop()

res = st.session_state["res"]
model, y, proba, feats = res["model"], res["y"], res["proba"], res["feats"]
cfg = res["cfg"]

st.caption(f"**{res['name']}** · {'LightGBM' if cfg['model'] == 'lgbm' else 'HistGradientBoosting'} · "
           f"{'KNN feature (k=' + str(cfg['k']) + ')' if model.knn_columns is not None else 'no KNN feature'} · "
           f"{len(feats)} features · train {res['n_train']:,} ({res['pos_train']} bankrupt) · "
           f"test {len(y):,} ({int(y.sum())} bankrupt) · trained in {res['seconds']:.1f}s")

# threshold controls (live: no retraining needed)
beta = st.number_input("β for F-beta (higher = recall matters more)", 0.1, 10.0, 2.0, 0.1)
sw = sweep(y, proba, beta)


def apply_rule(rule, target):
    if rule == "Max F1":
        t = sw.loc[sw["F1"].idxmax(), "threshold"]
    elif rule == "Max F-beta":
        t = sw.loc[sw["F-beta"].idxmax(), "threshold"]
    else:  # highest threshold that keeps recall ≥ target
        ok = sw[sw["Recall"] >= target]
        t = ok["threshold"].max() if len(ok) else 0.0
    st.session_state["thr_live"] = float(t)


tc1, tc2 = st.columns([3, 2])
thr = tc1.slider("Decision threshold (flag as bankrupt when probability ≥ this)", 0.0, 1.0,
                 step=0.005, key="thr_live")
with tc2:
    r1, r2, r3 = st.columns([2, 1.2, 1])
    rule = r1.selectbox("Suggest by", ["Max F1", "Max F-beta", "Recall ≥ target"], label_visibility="visible")
    tgt = r2.number_input("Target", 0.5, 0.99, 0.85, 0.01)
    r3.button("Apply", on_click=apply_rule, args=(rule, tgt), width="stretch")

m = metrics_at(y, proba, thr, beta)

row1 = st.columns(4)
row1[0].metric("Recall (sensitivity)", pct(m["recall"]),
               help="Share of actually-bankrupt companies that were flagged.")
row1[1].metric("Precision", pct(m["precision"]),
               delta=f"{m['precision'] / m['prevalence']:.1f}× base rate", delta_color="off",
               help="Share of flagged companies that really went bankrupt.")
row1[2].metric("F1", f"{m['f1']:.3f}")
row1[3].metric(f"F-beta (β={beta:g})", f"{m['fbeta']:.3f}")
row2 = st.columns(4)
row2[0].metric("PR AUC", f"{m['pr_auc']:.3f}", delta=f"baseline {m['prevalence']:.3f}", delta_color="off",
               help="Average precision from predicted probabilities; threshold-independent.")
row2[1].metric("ROC AUC", f"{m['roc_auc']:.3f}", help="Threshold-independent.")
row2[2].metric("Specificity", pct(m["specificity"]), help="Share of healthy companies correctly left alone.")
row2[3].metric("Balanced accuracy", pct(m["bal_acc"]))

tab_res, tab_curves, tab_rows, tab_new, tab_all = st.tabs(
    ["📊 Results", "📈 Curves & thresholds", "🔎 Test-set predictions", "📤 Predict new data", "🗓️ All five files"])

# ── results
with tab_res:
    left, right = st.columns([1, 1])
    with left:
        z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
        fig = go.Figure(go.Heatmap(z=z, x=["Predicted healthy", "Predicted bankrupt"],
                                   y=["Actually healthy", "Actually bankrupt"], colorscale="Blues",
                                   showscale=False, texttemplate="%{z}", textfont={"size": 22}))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(title="Confusion matrix", height=360, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("#### In plain terms")
        st.markdown(
            f"- Of **{m['tp'] + m['fn']}** bankrupt companies in the test set, the model caught **{m['tp']}** "
            f"and missed **{m['fn']}**.\n"
            f"- It flagged **{m['flagged']}** companies in total; **{m['tp']}** were right and "
            f"**{m['fp']}** were false alarms.\n"
            f"- Flagging at random would give ~{pct(m['prevalence'])} precision; this gives {pct(m['precision'])}.")
        imp = getattr(model.booster, "feature_importances_", None)
        if imp is not None:
            names = feats + (["knn_class1_fraction"] if model.knn_columns is not None else [])
            top = pd.Series(imp, index=names).sort_values(ascending=False).head(12)[::-1]
            fi = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h"))
            fi.update_layout(title="Top features (LightGBM split count)", height=330, margin=dict(t=50, b=10))
            st.plotly_chart(fi, width="stretch")
        else:
            st.caption("HistGradientBoosting has no built-in feature importances.")
    with st.expander("Why PR AUC here differs from your original script"):
        st.markdown(
            "`final_testing_file.py` calls `precision_recall_curve` on the **0/1 predictions**, which gives a "
            "curve with a single real point, so its PR AUC mostly reflects one precision/recall pair and "
            "changes with the threshold. This app computes PR AUC from the **predicted probabilities** "
            "(`average_precision_score`), which is the standard, threshold-independent version.")

# ── curves
with tab_curves:
    c1, c2 = st.columns(2)
    prec, rec, _ = precision_recall_curve(y, proba)
    f = go.Figure(go.Scatter(x=rec, y=prec, mode="lines", name="PR curve"))
    f.add_trace(go.Scatter(x=[m["recall"]], y=[m["precision"]], mode="markers", name="Current threshold",
                           marker=dict(size=13, color="crimson")))
    f.add_hline(y=m["prevalence"], line_dash="dot", annotation_text="base rate")
    f.update_layout(title=f"Precision–Recall (AP {m['pr_auc']:.3f})", xaxis_title="Recall",
                    yaxis_title="Precision", height=380, margin=dict(t=50, b=10))
    c1.plotly_chart(f, width="stretch")
    fpr, tpr, _ = roc_curve(y, proba)
    f = go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    f.add_trace(go.Scatter(x=[1 - m["specificity"]], y=[m["recall"]], mode="markers", name="Current threshold",
                           marker=dict(size=13, color="crimson")))
    f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot"), name="Chance"))
    f.update_layout(title=f"ROC (AUC {m['roc_auc']:.3f})", xaxis_title="False-positive rate",
                    yaxis_title="True-positive rate", height=380, margin=dict(t=50, b=10))
    c2.plotly_chart(f, width="stretch")

    f = go.Figure()
    for col in ["Recall", "Precision", "F1", "F-beta"]:
        f.add_trace(go.Scatter(x=sw["threshold"], y=sw[col], mode="lines", name=col))
    f.add_vline(x=thr, line_dash="dash", line_color="crimson")
    f.update_layout(title="Metrics vs threshold", xaxis_title="Threshold", height=380, margin=dict(t=50, b=10))
    st.plotly_chart(f, width="stretch")

    f = go.Figure()
    for cls, label in [(0, "Healthy"), (1, "Bankrupt")]:
        f.add_trace(go.Histogram(x=proba[y == cls], name=label, histnorm="probability", opacity=0.65,
                                 xbins=dict(start=0, end=1, size=0.02)))
    f.add_vline(x=thr, line_dash="dash", line_color="crimson")
    f.update_layout(barmode="overlay", title="Predicted probability by true class (each class sums to 100%)",
                    xaxis_title="Predicted probability", yaxis_title="Share of class", height=380,
                    margin=dict(t=50, b=10))
    st.plotly_chart(f, width="stretch")

# ── per-row predictions
with tab_rows:
    out = res["test"][feats].copy()
    out.insert(0, "actual_bankrupt", y)
    out.insert(1, "probability", proba)
    out.insert(2, "predicted_bankrupt", (proba >= thr).astype(int))
    out.insert(3, "outcome", np.select([(y == 1) & (proba >= thr), (y == 0) & (proba >= thr), (y == 1)],
                                       ["TP", "FP", "FN"], "TN"))
    pick = st.multiselect("Show outcomes", ["TP", "FP", "FN", "TN"], default=["TP", "FP", "FN"])
    view = out[out["outcome"].isin(pick)].sort_values("probability", ascending=False)
    st.dataframe(view, width="stretch", height=420,
                 column_config={"probability": st.column_config.ProgressColumn(
                     "probability", min_value=0.0, max_value=1.0, format="%.3f")})
    st.download_button("⬇️ Download test-set predictions", out.to_csv(index=False).encode(),
                       "test_predictions.csv", "text/csv")

# ── new data
with tab_new:
    st.write(f"Upload a CSV containing the **{len(feats)} features this model was trained on** "
             f"({'and an optional `class` column to score it' if True else ''}). Uses the current threshold "
             f"({thr:.3f}).")
    new = st.file_uploader("CSV to predict", type="csv", key="newdata")
    if new is not None:
        try:
            nd = pd.read_csv(new)
            miss = [c for c in feats if c not in nd.columns]
            if miss:
                st.error(f"Missing columns: {miss[:6]}{'…' if len(miss) > 6 else ''}")
            else:
                p_new = model.predict_proba(clean_features(nd[feats]).to_numpy(float))
                result = nd.copy()
                result["probability"] = p_new
                result["predicted_bankrupt"] = (p_new >= thr).astype(int)
                a, b = st.columns(2)
                a.metric("Companies scored", f"{len(result):,}")
                b.metric("Flagged as bankrupt", f"{int(result['predicted_bankrupt'].sum()):,}")
                if "class" in nd.columns:
                    yn = pd.to_numeric(nd["class"], errors="coerce")
                    ok = yn.isin([0, 1])
                    if ok.sum() and yn[ok].nunique() == 2:
                        mn = metrics_at(yn[ok].astype(int).to_numpy(), p_new[ok.to_numpy()], thr, beta)
                        st.write(f"Scored against `class`: recall **{pct(mn['recall'])}**, precision "
                                 f"**{pct(mn['precision'])}**, PR AUC **{mn['pr_auc']:.3f}**, "
                                 f"ROC AUC **{mn['roc_auc']:.3f}**. (If these rows were in training, "
                                 f"the numbers are optimistic.)")
                st.dataframe(result.sort_values("probability", ascending=False), width="stretch",
                             height=380)
                st.download_button("⬇️ Download predictions", result.to_csv(index=False).encode(),
                                   "predictions.csv", "text/csv")
        except Exception as e:  # noqa: BLE001
            st.error(f"Couldn't score that file: {e}")

# ── all files
with tab_all:
    if source != "Local folder":
        st.info("Switch the data source to *Local folder* to run all five files.")
    else:
        st.write("Re-trains with the **same settings** you last trained with on every file found in the folder "
                 "(each file uses its own starting threshold from the threshold setting, not the live slider).")
        if st.button("Run on all files"):
            rows, bar = [], st.progress(0.0)
            for n, i in enumerate(DATA_NAMES):
                p = Path(folder) / DATA_NAMES[i]
                if p.exists():
                    try:
                        r = run_experiment(p.read_bytes(), p.name, cfg)
                        mm = metrics_at(r["y"], r["proba"], r["model"].threshold, beta)
                        rows.append({"File": p.name, "Threshold": r["model"].threshold, "Test rows": len(r["y"]),
                                     "Bankrupt": int(r["y"].sum()), "Recall": mm["recall"],
                                     "Precision": mm["precision"], "F1": mm["f1"], "PR AUC": mm["pr_auc"],
                                     "ROC AUC": mm["roc_auc"]})
                    except Exception as e:  # noqa: BLE001
                        st.warning(f"{p.name}: {e}")
                bar.progress((n + 1) / len(DATA_NAMES))
            st.session_state["compare"] = pd.DataFrame(rows)
        if "compare" in st.session_state and len(st.session_state["compare"]):
            cmp_df = st.session_state["compare"]
            st.dataframe(cmp_df.style.format({"Threshold": "{:.3f}", "Recall": "{:.1%}", "Precision": "{:.1%}",
                                              "F1": "{:.3f}", "PR AUC": "{:.3f}", "ROC AUC": "{:.3f}"}),
                         width="stretch", hide_index=True)
            f = go.Figure()
            for col in ["Recall", "Precision", "PR AUC", "ROC AUC"]:
                f.add_trace(go.Bar(x=cmp_df["File"], y=cmp_df[col], name=col))
            f.update_layout(barmode="group", height=380, margin=dict(t=30, b=10))
            st.plotly_chart(f, width="stretch")
            st.download_button("⬇️ Download comparison", cmp_df.to_csv(index=False).encode(),
                               "all_files_metrics.csv", "text/csv")