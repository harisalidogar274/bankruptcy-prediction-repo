"""Data prep for the Polish bankruptcy CSVs (WINSORIZING version).

Same as data_prep.py, except outliers are handled by clipping each ratio to its
1st and 99th percentile instead of CHAID binning, so the two can be compared.
clean -> drop duplicates -> train/test split -> winsorize -> robust scale.
Everything that is learned (percentiles, median, IQR) is learned on TRAIN only, then applied to TEST.

The cleaning, duplicate removal and split are identical to data_prep.py (same SEED), so both
versions produce the SAME train and test rows; only the feature values differ.

Usage:
    from data_prep_winsor import prepped
    train, test = prepped["csv_result-1year.csv"]      # each has the 6 features + "class"
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
##########
FEATURES = ["Attr1", "Attr3", "Attr6", "Attr8", "Attr9", "Attr46","Attr22", "Attr42", "Attr25", "Attr15", "Attr40", "Attr52"] #FEATURES decided based on evalution of feature sets.
PATHS = [f"csv_result-{i}year.csv" for i in range(1, 6)]   # edit to your file names
TEST_SIZE = 0.2      # share of each file held out for testing
SEED = 42            # makes the split repeatable
LOWER, UPPER = 0.01, 0.99   # winsorizing percentiles


def prep(path):
    """Return (train, test) for one CSV, fully prepared."""
    # 1. load only the class + feature columns; the header row becomes the column names,
    #    so no text row is left in the data
    df = pd.read_csv(path, usecols=FEATURES + ["class"])

    # 2. everything to float64; '?' (and any other text) becomes NaN, inf becomes NaN
    df = df.apply(pd.to_numeric, errors="coerce").astype("float64")
    df = df.replace([np.inf, -np.inf], np.nan)

    # 3. drop rows with any missing value, then drop duplicate rows
    #    (duplicates go BEFORE the split so a copy cannot land in both train and test)
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    # 4. train/test split, keeping the bankrupt share equal in both parts
    train, test = train_test_split(df, test_size=TEST_SIZE, stratify=df["class"], random_state=SEED)
    train, test = train.copy(), test.copy()

    # 5. winsorize (outlier handling): find the 1st/99th percentile of each ratio on train,
    #    then cap train AND test values at those limits
    low, high = train[FEATURES].quantile(LOWER), train[FEATURES].quantile(UPPER)
    train[FEATURES] = train[FEATURES].clip(low, high, axis=1)
    test[FEATURES] = test[FEATURES].clip(low, high, axis=1)

    # 6. robust scale: median/IQR learned on train only
    scaler = RobustScaler().fit(train[FEATURES])
    train[FEATURES] = scaler.transform(train[FEATURES])
    test[FEATURES] = scaler.transform(test[FEATURES])

    # 7. label back to 0/1 integers
    train["class"], test["class"] = train["class"].astype(int), test["class"].astype(int)

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