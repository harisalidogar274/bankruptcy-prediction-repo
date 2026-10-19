import pandas as pd
from sklearn.metrics import accuracy_score
#DecisonTreeClassifier Acccuracy

from decision_tree_model import model
from decision_tree_model import features
from decision_tree_model import data_frame_1
from decision_tree_model import data_frame_2

predictions_DTC = model.predict(data_frame_2[features])
accuracy_score_DTC=accuracy_score(data_frame_2["class"],predictions_DTC)
print(accuracy_score_DTC)

#KNN Accuracy

from KNN_model import KNN


KNN_model_1=KNN()
KNN_model_1.fit(data_frame_1[features].to_numpy(), data_frame_1["class"].to_numpy())##by passing each as np.array we can bypass the need to make them numpy arrays in the KNN model code.
predictions_KNN=KNN_model_1.predict(data_frame_2[features].to_numpy())
accuracy_score_KNN=accuracy_score(data_frame_2["class"].to_numpy(),predictions_KNN)
print(accuracy_score_KNN)

#Linear Regression Accuracy

from Linear_Regression_model import LinearRegression

LR_model = LinearRegression()

LR_model.fit(data_frame_1[features].to_numpy(), data_frame_1["class"].to_numpy())
predictions_LR=LR_model.predict(data_frame_2[features].to_numpy())
accuracy_score_LR=accuracy_score(data_frame_2["class"].to_numpy(),predictions_LR)
print(accuracy_score_LR)

#Z-score (Altman Z' + gradient boosting) Accuracy

from bankruptcy_predictor import fit_final_model, predict_labels

Z_model, Z_threshold = fit_final_model(data_frame_1, data_frame_1["class"])
predictions_Z = predict_labels(Z_model, Z_threshold, data_frame_2)
accuracy_score_Z = accuracy_score(data_frame_2["class"], predictions_Z)
print(accuracy_score_Z)

#Comparison

scores = {"KNN": accuracy_score_KNN, "DTC": accuracy_score_DTC,
          "LR": accuracy_score_LR, "Z-score model": accuracy_score_Z}
winner = max(scores, key=scores.get)
print(f"{winner} wins with {scores[winner]:.4f}")

