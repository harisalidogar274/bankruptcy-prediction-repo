"""Data prep for the Polish bankruptcy CSVs.

Works on any CSV with the same columns (any number of rows). For each file it:
clean -> drop duplicates -> train/test split -> CHAID binning -> standardise.
Everything that is learned (bins, mean, std) is learned on TRAIN only, then applied to TEST.

Usage:
    from data_prep import prepped
    train, test = prepped["csv_result-1year.csv"]      # each has the 6 features + "class"
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["Attr1", "Attr3", "Attr6", "Attr8", "Attr9", "Attr46"]
PATHS = [f"csv_result-{i}year.csv" for i in range(1, 6)]   # edit to your file names
TEST_SIZE = 0.2      # share of each file held out for testing
SEED = 42            # makes the split repeatable
START_BINS = 10      # CHAID starts from this many quantile bins, then merges
ALPHA = 0.05         # neighbouring bins are merged unless their class mix differs at this level


def chaid_edges(x, y):
    """CHAID merge step for one feature: return the bin edges that survive.

    Start with quantile bins, then repeatedly merge the two neighbouring bins whose
    bankrupt/healthy mix is the LEAST different (chi-square test), until every
    neighbouring pair differs significantly.
    """
    edges = np.unique(np.quantile(x, np.linspace(0, 1, START_BINS + 1)[1:-1]))
    while len(edges) > 0:
        bins = np.digitize(x, edges)
        counts = np.array([[np.sum((bins == b) & (y == 0)), np.sum((bins == b) & (y == 1))]
                           for b in range(len(edges) + 1)])
        pairs = [counts[b:b + 2] for b in range(len(edges))]
        # a pair with no bankrupt (or no healthy) firms cannot be tested -> treat as "same"
        p = [1.0 if pair.sum(axis=0).min() == 0 else chi2_contingency(pair)[1] for pair in pairs]
        worst = int(np.argmax(p))
        if p[worst] <= ALPHA:
            break
        edges = np.delete(edges, worst)          # merge the least different neighbours
    return edges


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

    # 5. CHAID binning (outlier handling): learn edges on train, apply to train and test.
    #    A value beyond the training range falls into the first or last bin, so extremes are absorbed.
    edges = {f: chaid_edges(train[f].to_numpy(), train["class"].to_numpy()) for f in FEATURES}
    train, test = train.copy(), test.copy()
    for f in FEATURES:
        train[f] = np.digitize(train[f], edges[f])
        test[f] = np.digitize(test[f], edges[f])

    # 6. standardise the bin codes: mean/std learned on train only
    scaler = StandardScaler().fit(train[FEATURES])
    train[FEATURES] = scaler.transform(train[FEATURES])
    test[FEATURES] = scaler.transform(test[FEATURES])

    # 7. label back to 0/1 integers
    train["class"], test["class"] = train["class"].astype(int), test["class"].astype(int)

    print(f"{path}: kept {len(df)} rows | train {len(train)} ({train['class'].sum()} bankrupt) | "
          f"test {len(test)} ({test['class'].sum()} bankrupt) | bins per feature "
          f"{ {f: len(edges[f]) + 1 for f in FEATURES} }")
    return train.reset_index(drop=True), test.reset_index(drop=True)


# run on every file; prepped["<file name>"] = (train, test)
prepped = {}
for _path in PATHS:
    try:
        prepped[_path] = prep(_path)
    except FileNotFoundError:
        print(f"{_path}: not found, skipped")