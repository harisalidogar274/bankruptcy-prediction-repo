"""Data prep for the Polish bankruptcy CSVs (LightGBM + HistGradientBoosting version).

Both models are tree-based: they only use the order of values within a column and handle missing
values themselves, so nothing is winsorized, scaled or filled in.
clean -> drop duplicates -> train/test split -> class weights for TRAIN.
Both models get the SAME rows, split and weights, so any difference in results comes from the model.

Rows with missing values are KEPT (the boosters use them, and a missing ratio can itself be informative).
Same SEED and split rule as the other prep files, but the rows differ because none are dropped.

Usage:
    from data_prep_winsor import prepped
    train, test = prepped["csv_result-1year.csv"]      # each has the 64 ratios + "class"; train also has "weight"
    lgbm.fit(train[FEATURES], train["class"], sample_weight=train["weight"])   # LightGBM
    hgb.fit(train[FEATURES], train["class"], sample_weight=train["weight"])    # HistGradientBoostingClassifier
    # do NOT also set class_weight / scale_pos_weight / is_unbalance on the models (that would count twice)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
##########
FEATURES = [f"Attr{i}" for i in range(1, 65)] #all 64 ratios: the boosted trees choose which ones they use.
#FEATURES = [f"Attr{i}" for i in range(1, 65) if i != 27]
#FEATURES = [f"Attr{i}" for i in range(1, 65) if i != 27 and i != 11]
PATHS = [f"csv_result-{i}year.csv" for i in range(1, 6)]   # edit to your file names
TEST_SIZE = 0.2      # share of each file held out for testing
SEED = 42            # makes the split repeatable


def prep(path):
    """Return (train, test) for one CSV, fully prepared."""
    # 1. load only the class + feature columns; the header row becomes the column names,
    #    so no text row is left in the data
    df = pd.read_csv(path, usecols=FEATURES + ["class"])

    # 2. everything to float64; '?' (and any other text) becomes NaN, inf becomes NaN (XGBoost rejects inf)
    df = df.apply(pd.to_numeric, errors="coerce").astype("float64")
    df = df.replace([np.inf, -np.inf], np.nan)

    # 3. drop duplicate rows; rows with missing values are KEPT (the boosters handle NaN natively)
    #    (duplicates go BEFORE the split so a copy cannot land in both train and test)
    df = df.drop_duplicates().reset_index(drop=True)

    # 4. train/test split, keeping the bankrupt share equal in both parts
    train, test = train_test_split(df, test_size=TEST_SIZE, stratify=df["class"], random_state=SEED)
    train, test = train.copy(), test.copy()

    # 5. label back to 0/1 integers
    train["class"], test["class"] = train["class"].astype(int), test["class"].astype(int)

    # 6. imbalance: balanced per-row weight for TRAIN (a bankrupt company counts more, a healthy one less).
    #    Worked out from the train class counts only; test stays unweighted so its metrics are honest.
    train["weight"] = train["class"].map(len(train) / (2 * train["class"].value_counts()))

    print(f"{path}: kept {len(df)} rows | train {len(train)} ({train['class'].sum()} bankrupt) | "
          f"test {len(test)} ({test['class'].sum()} bankrupt)")
    return train.reset_index(drop=True), test.reset_index(drop=True)


# run on every file; prepped["<file name>"] = (train, test)
prepped = {}
for _path in PATHS:
    try:
        prepped[_path] = prep(_path)
    except FileNotFoundError:
        print(f"{_path}: not found, skipped")