"""Data cleaning + preparation for the Polish bankruptcy files.

Same set-up as decision_tree_model.py:
    data_frame_1 = csv_result-1year.csv  -> TRAINING data
    data_frame_2 = csv_result-2year.csv  -> TESTING data
    features     = ["Attr1", "Attr7", "Attr14", "Attr18"]

What this file exports (import them from comparison.py):
    features, data_frame_1, data_frame_2
        Tidy frames: '?' -> NaN, every Attr column is float64, "class" is int (0/1).
        ALL columns and ALL rows are kept (no dropna), values are NOT scaled, because the
        Altman Z' model needs the raw ratios.
    data_frame_1_prepared, data_frame_2_prepared
        Only the `features` columns, clipped + gaps filled + scaled (float64, no NaN),
        with the SAME index as data_frame_1 / data_frame_2. Use these for DTC, KNN and LR.

Run it directly (python data_prep.py) for a data-quality report.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

features = ["Attr1", "Attr9", "Attr3", "Attr6", "Attr8", "Attr46"]
PATH_1 = "csv_result-1year.csv"
PATH_2 = "csv_result-2year.csv"


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip every column to its 1st/99th percentile, learned on TRAINING data only."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.lo_ = np.nanpercentile(X, 1, axis=0)
        self.hi_ = np.nanpercentile(X, 99, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=np.float64), self.lo_, self.hi_)


def clean_data_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types and remove duplicates. NO rows are dropped for missing values.

    - '?' becomes NaN, then every Attr column is converted to float64. (Before, a single
      '?' left the whole column as text, which is why KNN needed .astype(float).)
    - "class" becomes int, so every metric compares int labels with int predictions.
    - exact duplicate rows (ignoring `id`) are removed.
    """
    df = df.replace("?", np.nan)
    attr_columns = [c for c in df.columns if c.startswith("Attr")]
    df[attr_columns] = df[attr_columns].apply(pd.to_numeric, errors="coerce").astype(np.float64)
    df = df.dropna(subset=["class"])
    df["class"] = df["class"].astype(int)
    return df.drop_duplicates(subset=attr_columns + ["class"]).reset_index(drop=True)


# ---------------------------------------------------------------- load + clean
data_frame_1 = clean_data_frame(pd.read_csv(PATH_1))
data_frame_2 = clean_data_frame(pd.read_csv(PATH_2))

# a company row that is identical in BOTH files would let the models "see" the test answer
_attrs = [c for c in data_frame_1.columns if c.startswith("Attr")]
_shared = (data_frame_2[_attrs]
           .merge(data_frame_1[_attrs].drop_duplicates(), on=_attrs, how="left", indicator=True)
           ["_merge"].to_numpy() == "both")
data_frame_2 = data_frame_2.loc[~_shared].reset_index(drop=True)

# ---------------------------------------------------------------- numeric prep
# Fitted on data_frame_1 (training) ONLY, then applied to data_frame_2 (testing).
# Why: clipping stops extreme outliers dominating, filling the median keeps every row
# (dropna threw away most of the data), scaling puts the columns on equal footing,
# which KNN (distances) and the gradient-descent regression both need.
_prep = Pipeline([("clip", Winsorizer()),
                  ("impute", SimpleImputer(strategy="median")),
                  ("scale", StandardScaler())])

data_frame_1_prepared = pd.DataFrame(_prep.fit_transform(data_frame_1[features]),
                                     columns=features, index=data_frame_1.index)
data_frame_2_prepared = pd.DataFrame(_prep.transform(data_frame_2[features]),
                                     columns=features, index=data_frame_2.index)
data_frame_1_prepared["class"] = data_frame_1["class"]
data_frame_2_prepared["class"] = data_frame_2["class"]


if __name__ == "__main__":
    raw_1 = pd.read_csv(PATH_1).replace("?", np.nan)
    raw_2 = pd.read_csv(PATH_2).replace("?", np.nan)
    print(f"{'':28s}{'train (1year)':>15s}{'test (2year)':>15s}")
    print(f"{'rows in file':28s}{len(raw_1):>15d}{len(raw_2):>15d}")
    print(f"{'rows left by old dropna':28s}{len(raw_1.dropna()):>15d}{len(raw_2.dropna()):>15d}")
    print(f"{'rows kept now':28s}{len(data_frame_1):>15d}{len(data_frame_2):>15d}")
    print(f"{'bankrupt (class = 1)':28s}{int(data_frame_1['class'].sum()):>15d}"
          f"{int(data_frame_2['class'].sum()):>15d}")
    print(f"{'bankruptcy rate':28s}{data_frame_1['class'].mean():>15.2%}"
          f"{data_frame_2['class'].mean():>15.2%}")
    print(f"{'missing in the 4 features':28s}{int(data_frame_1[features].isna().sum().sum()):>15d}"
          f"{int(data_frame_2[features].isna().sum().sum()):>15d}")
    print("dtypes OK:", all(data_frame_1[features].dtypes == np.float64),
          data_frame_1['class'].dtype, "| NaN left in prepared data:",
          int(data_frame_1_prepared.isna().sum().sum() + data_frame_2_prepared.isna().sum().sum()))