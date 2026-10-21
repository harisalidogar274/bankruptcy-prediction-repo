from data_prep_winsor import prepped as prepped_winsor
from data_prep_winsor import FEATURES as FEATURES_winsor
from clean_KNN_model import KNNBoost
from sklearn.metrics import auc, precision_recall_curve, recall_score, precision_score

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