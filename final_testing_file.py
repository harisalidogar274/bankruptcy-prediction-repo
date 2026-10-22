from data_prep_LGBM_HGB import prepped as prepped_winsor
from data_prep_LGBM_HGB import FEATURES as FEATURES_winsor
from Final_model import KNNBoost
from sklearn.metrics import auc, precision_recall_curve, recall_score, precision_score

#Winsor model performance
file = int(input("please choose the file to make predictions for"))
train_winsor, test_winsor = prepped_winsor[f"csv_result-{file}year.csv"] 
model_KNN_winsor = KNNBoost(model="lgbm")
model_KNN_winsor.fit(train_winsor[FEATURES_winsor], train_winsor["class"], sample_weight=train_winsor["weight"])
predictions_winsor = model_KNN_winsor.predict(test_winsor[FEATURES_winsor])
#####metrics
sensitivity_score_winsor=recall_score(test_winsor['class'], predictions_winsor)
precision_winsor, recall_winsor, thresholds_winsor = precision_recall_curve(test_winsor['class'], predictions_winsor)
pr_auc_winsor = auc(recall_winsor, precision_winsor)
precision_score_val_winsor = precision_score(test_winsor['class'], predictions_winsor)
print(f"PR AUC is {pr_auc_winsor}, and Recall/Sensitivity is {sensitivity_score_winsor} and Precision is {precision_score_val_winsor}")