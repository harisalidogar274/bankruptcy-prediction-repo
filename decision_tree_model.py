import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
data_frame_1 = pd.read_csv("csv_result-1year.csv")
data_frame_2 = pd.read_csv("csv_result-2year.csv")
features=["Attr1", "Attr9", "Attr3", "Attr6", "Attr8", "Attr46"]
data_frame_1.replace('?',np.nan,inplace=True)
data_frame_2.replace('?',np.nan,inplace=True)
data_frame_1.dropna(inplace=True)
data_frame_2.dropna(inplace=True)
model = DecisionTreeClassifier(max_depth=5).fit(data_frame_1[features],data_frame_1["class"])
predictions=model.predict(data_frame_2[features])