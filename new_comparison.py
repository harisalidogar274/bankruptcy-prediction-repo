import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             fbeta_score, precision_score, recall_score)

# All models now read the SAME cleaned data (see data_prep.py).
#   data_frame_1 / data_frame_2                   -> tidy frames, no rows dropped, unscaled
#   data_frame_1_prepared / data_frame_2_prepared -> the 4 features, clipped + filled + scaled
from data_prep import features
from data_prep import data_frame_1
from data_prep import data_frame_2
from data_prep import data_frame_1_prepared
from data_prep import data_frame_2_prepared

y_true = data_frame_2["class"].to_numpy().astype(int)   # int 0/1 labels for every metric below

#DecisonTreeClassifier Acccuracy

from decision_tree_model import model

model.fit(data_frame_1_prepared[features], data_frame_1["class"])   # same settings (max_depth=5), refit on the prepared data
predictions_DTC = model.predict(data_frame_2_prepared[features]).astype(int)
accuracy_score_DTC=accuracy_score(y_true,predictions_DTC)
print(accuracy_score_DTC)

#KNN Accuracy

from KNN_model import KNN


KNN_model_1=KNN()
KNN_model_1.fit(data_frame_1_prepared[features].to_numpy(), data_frame_1["class"].to_numpy())##by passing each as np.array we can bypass the need to make them numpy arrays in the KNN model code.
predictions_KNN=KNN_model_1.predict(data_frame_2_prepared[features].to_numpy()).astype(int)   # KNN returns bool -> int so it matches the int labels
accuracy_score_KNN=accuracy_score(y_true,predictions_KNN)
print(accuracy_score_KNN)

#Linear Regression Accuracy

from Linear_Regression_model import LinearRegression

LR_model = LinearRegression()

LR_model.fit(data_frame_1_prepared[features].to_numpy(), data_frame_1["class"].to_numpy())
predictions_LR=LR_model.predict(data_frame_2_prepared[features].to_numpy()).astype(int)
accuracy_score_LR=accuracy_score(y_true,predictions_LR)
print(accuracy_score_LR)

#Z-score (Altman Z' + gradient boosting) Accuracy

from bankruptcy_predictor import fit_final_model, predict_labels

# This model needs the RAW ratios (Z' uses Attr3, 6, 7, 8, 9), so it gets the unscaled frames.
Z_model, Z_threshold = fit_final_model(data_frame_1, data_frame_1["class"])
predictions_Z = np.asarray(predict_labels(Z_model, Z_threshold, data_frame_2)).astype(int)
accuracy_score_Z = accuracy_score(y_true, predictions_Z)
print(accuracy_score_Z)

#Baseline: predict "not bankrupt" for every company (the accuracy trap, for reference only)

predictions_baseline = np.zeros_like(y_true)
accuracy_score_baseline = accuracy_score(y_true, predictions_baseline)

#Comparison

all_predictions = {"KNN": predictions_KNN, "DTC": predictions_DTC,
                   "LR": predictions_LR, "Z-score model": predictions_Z,
                   "Baseline (predict nobody)": predictions_baseline}

rows = {}
for name, p in all_predictions.items():
    tn, fp, fn, tp = confusion_matrix(y_true, p, labels=[0, 1]).ravel()
    rows[name] = {"Accuracy": accuracy_score(y_true, p),
                  "Balanced acc": balanced_accuracy_score(y_true, p),
                  "Recall (caught)": recall_score(y_true, p, zero_division=0),
                  "Precision": precision_score(y_true, p, zero_division=0),
                  "F2": fbeta_score(y_true, p, beta=2, zero_division=0),
                  "Caught": int(tp), "Missed": int(fn), "False alarms": int(fp)}
table = pd.DataFrame(rows).T
for column in ["Caught", "Missed", "False alarms"]:
    table[column] = table[column].astype(int)

print(f"\nTest set: {len(y_true)} companies, {int(y_true.sum())} bankrupt ({y_true.mean():.1%})")
print(table.round(3).sort_values("Balanced acc", ascending=False).to_string())

scores = {"KNN": accuracy_score_KNN, "DTC": accuracy_score_DTC,
          "LR": accuracy_score_LR, "Z-score model": accuracy_score_Z}
balanced_scores = {name: table.loc[name, "Balanced acc"] for name in scores}

winner = max(balanced_scores, key=balanced_scores.get)
print(f"\n{winner} wins with balanced accuracy {balanced_scores[winner]:.4f}")
print(f"(best plain accuracy: {max(scores, key=scores.get)} {max(scores.values()):.4f}; "
      f"'predict nobody' alone scores {accuracy_score_baseline:.4f})")
for name in scores:
    if table.loc[name, "Caught"] == 0:
        print(f"WARNING: {name} caught 0 bankruptcies, its accuracy only reflects the base rate.")