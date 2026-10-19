import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score,roc_curve)

# All models now read the SAME cleaned data (see data_prep.py).
#   data_frame_1 / data_frame_2                   -> tidy frames, no rows dropped, unscaled
#   data_frame_1_prepared / data_frame_2_prepared -> the 4 features, clipped + filled + scaled
from data_prep import features
from data_prep import data_frame_1
from data_prep import data_frame_2


from KNN_model import KNN,KNNBoost
from sklearn.metrics import auc, precision_recall_curve, recall_score, precision_score
##############################
##model_KNN = KNN()
##model_KNN.fit(data_frame_1_prepared[features].values,data_frame_1_prepared["class"].values)
##predictions = model_KNN.predict(data_frame_2_prepared[features].values)

##print(f"Accuracy: {accuracy_score(data_frame_2_prepared['class'].values, predictions)}")
##print(auc_roc_score(data_frame_2_prepared['class'].values, predictions))#my auc_roc is balanced accuracy. (sensitivity+specificity)/2
##############################

from KNN_model import KNNBoost
all_attrs = [c for c in data_frame_1.columns if c.startswith("Attr") and c != "Attr36"]
model_KNN = KNNBoost()
model_KNN.fit(data_frame_1[all_attrs].values, data_frame_1["class"].values)
predictions = model_KNN.predict(data_frame_2[all_attrs].values)
#####print(auc_roc_score(data_frame_2['class'].values, predictions))
sensitivity_score=recall_score(data_frame_2['class'].values, predictions)
precision, recall, thresholds = precision_recall_curve(data_frame_2['class'].values, predictions)
pr_auc = auc(recall, precision)
precision_score_val = precision_score(data_frame_2['class'].values, predictions)
print(f"PR AUC is {pr_auc}, and Sensitivity is {sensitivity_score} and Precision is {precision_score_val}")

# WHY THIS KNN (k=50, flag if >=10% of neighbours are bankrupt) SCORED ROC-AUC ~0.54
# (measured with an equivalent KNN on random 70/30 splits of the 2year file, 3 splits)
#
# 1. The 0.54 is measured on hard 0/1 labels. ROC-AUC on 0/1 predictions collapses the
#    ROC curve to a single point, so it equals balanced accuracy = (recall + specificity) / 2.
#    The same neighbour shares used as a continuous score give ~0.66 AUC. predict() throws
#    that ranking away by returning only 0 or 1.
#
# 2. The 10% cutoff is too close to chance. The base rate is ~4%. With k=50, a company
#    with no bankruptcy signal at all has a ~5% chance of getting >=5 bankrupt neighbours
#    (a binomial fluke), so ~145 of ~2,900 healthy test companies get flagged by luck alone.
#    We flagged ~300 companies (10% of the test set) to catch only ~17% of the 119 bankrupt
#    ones. Precision was ~6%, only ~1.5x the base rate, so about 94% of flags were false alarms.
#
# 3. The neighbourhoods carry only a weak signal. Each feature alone separates bankrupt from
#    healthy only weakly (single-feature ROC-AUC ~0.58-0.63), and bankruptcy here is a modest
#    shift in the odds, not a clear cluster. A 50-company neighbourhood is also wide enough
#    that its bankrupt share (std ~0.03 at a 4% rate) is noisy next to a 10% cutoff.
#
# 4. Redundant features. Attr1 and Attr7 correlate 0.99 (both profitability) and Attr8 and
#    Attr2 correlate -1.00 (both leverage), so the 6 features carry only 4 distinct signals,
#    and profitability and leverage are double-weighted in the distance.
#
# 5. KNN cannot learn feature importance. Euclidean distance weights every feature equally, so
#    it cannot down-weight the noisy ones or combine them non-linearly the way gradient
#    boosting does (~0.95 AUC on this data).