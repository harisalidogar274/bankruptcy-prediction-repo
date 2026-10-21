import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score,roc_curve)
from data_prep_CHAID import prepped as prepped_CHAID
from data_prep_CHAID import FEATURES as FEATURES_CHAID
from data_prep_winsor import prepped as prepped_winsor
from data_prep_winsor import FEATURES as FEATURES_winsor
from KNN_model import KNN,KNNBoost
from sklearn.metrics import auc, precision_recall_curve, recall_score, precision_score
from KNN_model import KNNBoost

#CHAID model performance
train_CHAID, test_CHAID = prepped_CHAID["csv_result-1year.csv"] 
model_KNN_CHAID = KNNBoost()
model_KNN_CHAID.fit(train_CHAID[FEATURES_CHAID],train_CHAID["class"])
predictions_CHAID = model_KNN_CHAID.predict(test_CHAID[FEATURES_CHAID])
#####metrics
sensitivity_score_CHAID=recall_score(test_CHAID['class'], predictions_CHAID)
precision_CHAID, recall_CHAID, thresholds_CHAID = precision_recall_curve(test_CHAID['class'], predictions_CHAID)
pr_auc_CHAID = auc(recall_CHAID, precision_CHAID)
precision_score_val_CHAID = precision_score(test_CHAID['class'], predictions_CHAID)
print(f" CHAID: \n PR AUC is {pr_auc_CHAID}, and Sensitivity is {sensitivity_score_CHAID} and Precision is {precision_score_val_CHAID}")


#Winsor model performance
train_winsor, test_winsor = prepped_winsor["csv_result-1year.csv"] 
model_KNN_winsor = KNNBoost()
model_KNN_winsor.fit(train_winsor[FEATURES_winsor],train_winsor["class"])
predictions_winsor = model_KNN_winsor.predict(test_winsor[FEATURES_winsor])
#####metrics
sensitivity_score_winsor=recall_score(test_winsor['class'], predictions_winsor)
precision_winsor, recall_winsor, thresholds_winsor = precision_recall_curve(test_winsor['class'], predictions_winsor)
pr_auc_winsor = auc(recall_winsor, precision_winsor)
precision_score_val_winsor = precision_score(test_winsor['class'], predictions_winsor)
print(f" Winsor: \n PR AUC is {pr_auc_winsor}, and Sensitivity is {sensitivity_score_winsor} and Precision is {precision_score_val_winsor}")