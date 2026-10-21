import pandas as pd
from data_prep_winsor import prepped as prepped_winsor
from data_prep_winsor import FEATURES as FEATURES_winsor
from KNN_model import KNNBoost
from sklearn.metrics import auc, precision_recall_curve, recall_score, precision_score
from KNN_model import KNNBoost

"""Candidate feature sets to compare for the KNN + gradient boosting model.

Source: Zieba, Tomczak & Tomczak (2016), Table 4 (top-20 features per year), Table 5 (synthetic
features) and section 3.4. Column names follow the CSVs: paper "X25" = column "Attr25".

Attr27 is deliberately EXCLUDED from every set: on the 1-year file it is missing for 44% of
bankrupt firms but only 3% of healthy ones, so dropping incomplete rows would remove almost half
of the bankrupt firms. Test it separately (median fill + a "missing" flag) rather than via dropna.

How to use (so every set is scored on the SAME rows and the SAME split):
    1. In data_prep_winsor.py set FEATURES = ALL_COLUMNS, so the cleaning, de-duplication and
       train/test split are done ONCE on the union of every column below.
    2. For each set, score the model on train[cols] / test[cols] only.
Preparing each set separately would change which rows survive dropna/drop_duplicates and
therefore the train/test rows, making the metrics not comparable.
"""


def union(*lists):
    """Combine column lists, keeping order and removing repeats."""
    return list(dict.fromkeys(col for cols in lists for col in cols))


# A. current set (6)
BASELINE = ["Attr1", "Attr3", "Attr6", "Attr8", "Attr9", "Attr46"]

# B. features in the paper's top 20 in ALL five years (3)
CORE3 = ["Attr25", "Attr40", "Attr52"]

# C. one per ratio family, from the paper's most consistent features (6)
#    X22 profitability on assets, X42 profitability on sales, X25 equity share,
#    X15 leverage, X40 liquidity, X52 liabilities turnover
FAMILY6 = ["Attr22", "Attr42", "Attr25", "Attr15", "Attr40", "Attr52"]

# G. features in the paper's top 20 in at least 4 of 5 years, minus Attr27 (13)
CONSISTENT13 = CORE3 + ["Attr13", "Attr22", "Attr31", "Attr42", "Attr15", "Attr9",
                        "Attr36", "Attr48", "Attr5", "Attr58"]

CONFIGS = {
    "A_baseline": BASELINE,                              # 6
    "B_core3": CORE3,                                    # 3
    "C_family6": FAMILY6,                                # 6
    "E_baseline+core3": union(BASELINE, CORE3),          # 9
    "F_baseline+family6": union(BASELINE, FAMILY6),      # 12
    "G_consistent13": CONSISTENT13,                      # 13
}

# D. top 6 of Table 4 for the matching year (6), skipping Attr27 and taking the next-ranked
#    feature instead. Only valid if your n-year file is the paper's n-th year file
#    (row counts 7,027 / 10,173 / 10,503 / 9,792 / 5,910).
PER_YEAR_TOP6 = {
    1: ["Attr16", "Attr52", "Attr32", "Attr28", "Attr5", "Attr40"],
    2: ["Attr40", "Attr15", "Attr5", "Attr25", "Attr36", "Attr22"],
    3: ["Attr15", "Attr22", "Attr52", "Attr40", "Attr5", "Attr25"],
    4: ["Attr22", "Attr52", "Attr15", "Attr25", "Attr40", "Attr58"],
    5: ["Attr25", "Attr22", "Attr15", "Attr52", "Attr53", "Attr14"],
}

# Synthetic features from Table 5 (1st year). Test ONE AT A TIME on top of the best set above.
# Evaluate with df.eval(expression). Needs Attr18, Attr29, Attr34, Attr47 loaded too, and divisions
# can give inf (treat as missing). S1 and S2 use Attr27, so they inherit its missing-data problem.
# The paper itself warns these were not validated individually and can overfit.
SYNTHETIC = {
    "S1": "(Attr22 * Attr25) + Attr27",
    "S2": "Attr47 / Attr27",
    "S3": "Attr18 - Attr34",
    "S4": "Attr29",                      # log of total assets (a raw ratio, but very prominent)
}

ALL_COLUMNS = union(*CONFIGS.values(), *PER_YEAR_TOP6.values())


# file to test, and which year of the paper it corresponds to (picks set D)
FILE = "csv_result-1year.csv"
YEAR = 1
CONFIGS["D_per_year_top6"] = PER_YEAR_TOP6[YEAR]

# same train/test rows for every feature set
train_winsor, test_winsor = prepped_winsor[FILE]

results = []   # one row of metrics per feature set
for name, FEATURES_winsor in CONFIGS.items():
    #Winsor model performance on this feature set
    model_KNN_winsor = KNNBoost()
    model_KNN_winsor.fit(train_winsor[FEATURES_winsor],train_winsor["class"])
    predictions_winsor = model_KNN_winsor.predict(test_winsor[FEATURES_winsor])
    #####metrics
    sensitivity_score_winsor=recall_score(test_winsor['class'], predictions_winsor)
    precision_winsor, recall_winsor, thresholds_winsor = precision_recall_curve(test_winsor['class'], predictions_winsor)
    pr_auc_winsor = auc(recall_winsor, precision_winsor)
    precision_score_val_winsor = precision_score(test_winsor['class'], predictions_winsor)
    # store this feature set's results
    results.append({"Feature set": name, "N features": len(FEATURES_winsor),
                    "PR AUC": pr_auc_winsor, "Sensitivity": sensitivity_score_winsor,
                    "Precision": precision_score_val_winsor})

# comparison table: one row per feature set
table = pd.DataFrame(results).set_index("Feature set")
print(table.round(4))

# winner, printed outside the table (PR AUC is the headline metric; change the column to switch)
winner = table["PR AUC"].idxmax()
print(f"\nWinner (highest PR AUC): {winner}")

