import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score,roc_curve)

# All models now read the SAME cleaned data (see data_prep.py).
#   data_frame_1 / data_frame_2                   -> tidy frames, no rows dropped, unscaled
#   data_frame_1_prepared / data_frame_2_prepared -> the 4 features, clipped + filled + scaled
from data_prep import features
from data_prep import data_frame_1_prepared
from data_prep import data_frame_2_prepared


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
all_attrs = [c for c in data_frame_1_prepared.columns if c.startswith("Attr") and c != "Attr36"]
model_KNN = KNNBoost()
model_KNN.fit(data_frame_1_prepared[all_attrs].values, data_frame_1_prepared["class"].values)
predictions = model_KNN.predict(data_frame_2_prepared[all_attrs].values)
#####print(auc_roc_score(data_frame_2['class'].values, predictions))
sensitivity_score=recall_score(data_frame_2_prepared['class'].values, predictions)
precision, recall, thresholds = precision_recall_curve(data_frame_2_prepared['class'].values, predictions)
pr_auc = auc(recall, precision)
precision_score_val = precision_score(data_frame_2_prepared['class'].values, predictions)
print(f"PR AUC is {pr_auc}, and Sensitivity is {sensitivity_score} and Precision is {precision_score_val}")