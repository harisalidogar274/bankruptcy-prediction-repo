
import numpy as np
# top of file, under "from collections import Counter":
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from collections import Counter
##I employed numpy arrays in this model and the code is compatible with those. The use of normal lists would still work granted the code was altered to be compatible to such an arrangement.
def euclidean_distance(x_a,x_b):

    x1 = x_a.astype(float)##this dealt with the problem of x_a and x_b being a string value and not workable here. 
    x2 = x_b.astype(float)
    distance = np.sqrt(np.sum((x1-x2)**2))
    ##this is basically the distance formula ((x2-x1)^2+(y2-y1)^2)^0.5
    return distance

def auc_roc_score(y_true, y_score):
    return roc_auc_score(np.asarray(y_true).astype(int), np.asarray(y_score, dtype=float))#balanced accuracy

class KNN:
    def __init__(self, k=50):
        self.k=k
        self.threshold=0.10
    def fit(self, X, y):#we fit the model with the training data
        self.X_train = X
        self.y_train = y
        self.find_optimal_threshold_beta()

    def find_optimal_threshold(self):
        #scores every TRAINING company by the share of class 1 among its k nearest OTHER training companies
        #(leave-one-out, so a company never counts as its own neighbour), then picks the cutoff that maximises
        #true positive rate - false positive rate (Youden's J = balanced accuracy) on the ROC curve.
        X = np.asarray(self.X_train, dtype=float)
        y = np.asarray(self.y_train).astype(int)
        squared_norms = np.sum(X**2, axis=1)
        class_1_proportions = np.empty(len(X))
        for start in range(0, len(X), 500):#chunks keep memory small; numpy does all the distances at once
            chunk = X[start:start+500]
            squared_distance = np.sum(chunk**2, axis=1)[:, None] + squared_norms[None, :] - 2 * chunk @ X.T
            squared_distance[np.arange(len(chunk)), np.arange(start, start+len(chunk))] = np.inf
            k_indices = np.argpartition(squared_distance, self.k - 1, axis=1)[:, :self.k]
            class_1_proportions[start:start+len(chunk)] = y[k_indices].mean(axis=1)
        fpr, tpr, thresholds = roc_curve(y, class_1_proportions)
        best_threshold = thresholds[np.argmax(tpr - fpr)]
        if np.isfinite(best_threshold) and best_threshold > 0:#safety: keep the old 10% if no usable cutoff exists
            self.threshold = float(best_threshold)
        return self.threshold
    
    def find_optimal_threshold_beta(self):
        beta = 3  # F-beta weight: beta > 1 favours recall (catching bankruptcies), beta < 1 favours precision, 1 = balanced
        #scores every TRAINING company by the share of class 1 among its k nearest OTHER training companies
        #(leave-one-out, so a company never counts as its own neighbour), then picks the cutoff that maximises
        #the F-beta score (precision and recall, with recall counted beta times as important as precision).
        X = np.asarray(self.X_train, dtype=float)
        y = np.asarray(self.y_train).astype(int)
        squared_norms = np.sum(X**2, axis=1)
        class_1_proportions = np.empty(len(X))
        for start in range(0, len(X), 500):#chunks keep memory small; numpy does all the distances at once
            chunk = X[start:start+500]
            squared_distance = np.sum(chunk**2, axis=1)[:, None] + squared_norms[None, :] - 2 * chunk @ X.T
            squared_distance[np.arange(len(chunk)), np.arange(start, start+len(chunk))] = np.inf
            k_indices = np.argpartition(squared_distance, self.k - 1, axis=1)[:, :self.k]
            class_1_proportions[start:start+len(chunk)] = y[k_indices].mean(axis=1)
        precision, recall, thresholds = precision_recall_curve(y, class_1_proportions)
        f_beta = (1 + beta**2) * precision * recall / np.maximum(beta**2 * precision + recall, 1e-12)
        best_threshold = thresholds[np.argmax(f_beta[:-1])]#last point has no threshold, so it is left out
        if np.isfinite(best_threshold) and best_threshold > 0:#safety: keep the old 10% if no usable cutoff exists
            self.threshold = float(best_threshold)
        return self.threshold

    def predict(self, X_test):# we predict for the test data using the training data
        #Applies the helper funciton(_predict) to each data point in the test set.
        ###
        predictions = [self._predict(np.array(x_test,dtype=float)) for x_test in X_test]
        return np.array(predictions).astype(bool)
        ###
        ##this fixed the problem arising from predictions not being a bool which resulted in accuracy_score not working properly



    def _predict(self, x_test):
        #takes a single data point value from testing values
        
        #computes the distance between the x_test value chosen and the rest of the X_test data points.
        distance = [euclidean_distance(x_test,np.array(x_train,dtype=float)) for x_train in self.X_train] ####################
        
        #gets the closest k neighbors.
        k_indices = np.argsort(distance)[:self.k]
        ##the [:self.k] after the application of argsort gives us the first k indices  
        ##np.argsort gives us a numpy array of index numbers with indices in order according to an array sorted in ascending order. so the first index number corressponds to the smallest value.
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # CUSTOMIZATION: 10% Cutoff Threshold for Class 1
        # Count how many of the k-nearest neighbors are actually class 1
        class_1_count = sum(1 for label in k_nearest_labels if label == 1)
        
        # Calculate the proportion (percentage) of class 1 neighbors
        class_1_proportion = class_1_count / self.k
        
        # If class 1 makes up 10% or more of the neighbors, predict 1
        if class_1_proportion >= self.threshold:
            return 1
        else:
            return 0
 


#########################################################################################################################
# ADDED: gradient boosting on top of the KNN code (nothing above this line was changed)
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

class KNNBoost(KNN):
    def __init__(self, k=50, knn_columns=None):
        super().__init__(k)
        ##knn_columns=None  -> pure gradient boosting on every column (best on the full feature set).
        ##knn_columns=[0,2] -> ALSO adds the share of class 1 among the k nearest neighbours (measured on
        ##                     those column positions, which must contain no NaN) as one extra feature.
        self.knn_columns = knn_columns
        self.booster = HistGradientBoostingClassifier(class_weight="balanced", learning_rate=0.05, max_iter=300,
                                                      max_leaf_nodes=15, min_samples_leaf=20,
                                                      l2_regularization=1.0, random_state=0)

    def _class_1_proportions(self, X, leave_one_out=False):
        #same neighbour share as find_optimal_threshold, for any rows X (numpy, chunked)
        train = self.X_train[:, self.knn_columns]
        X = np.asarray(X, dtype=float)[:, self.knn_columns]
        squared_norms = np.sum(train**2, axis=1)
        class_1_proportions = np.empty(len(X))
        for start in range(0, len(X), 500):
            chunk = X[start:start+500]
            squared_distance = np.sum(chunk**2, axis=1)[:, None] + squared_norms[None, :] - 2 * chunk @ train.T
            if leave_one_out:#a training company must not be its own neighbour
                squared_distance[np.arange(len(chunk)), np.arange(start, start+len(chunk))] = np.inf
            k_indices = np.argpartition(squared_distance, self.k - 1, axis=1)[:, :self.k]
            class_1_proportions[start:start+len(chunk)] = self.y_train[k_indices].mean(axis=1)
        return class_1_proportions

    def _with_knn_feature(self, X, leave_one_out=False):
        X = np.asarray(X, dtype=float)
        if self.knn_columns is None:
            return X
        return np.c_[X, self._class_1_proportions(X, leave_one_out)]

    def find_optimal_threshold_beta(self):
        beta = 4  # F-beta weight: beta > 1 favours recall (catching bankruptcies), beta < 1 favours precision, 1 = balanced
        #scores every TRAINING company by the share of class 1 among its k nearest OTHER training companies
        #(leave-one-out, so a company never counts as its own neighbour), then picks the cutoff that maximises
        #the F-beta score (precision and recall, with recall counted beta times as important as precision).
        X = np.asarray(self.X_train, dtype=float)
        y = np.asarray(self.y_train).astype(int)
        squared_norms = np.sum(X**2, axis=1)
        class_1_proportions = np.empty(len(X))
        for start in range(0, len(X), 500):#chunks keep memory small; numpy does all the distances at once
            chunk = X[start:start+500]
            squared_distance = np.sum(chunk**2, axis=1)[:, None] + squared_norms[None, :] - 2 * chunk @ X.T
            squared_distance[np.arange(len(chunk)), np.arange(start, start+len(chunk))] = np.inf
            k_indices = np.argpartition(squared_distance, self.k - 1, axis=1)[:, :self.k]
            class_1_proportions[start:start+len(chunk)] = y[k_indices].mean(axis=1)
        precision, recall, thresholds = precision_recall_curve(y, class_1_proportions)
        f_beta = (1 + beta**2) * precision * recall / np.maximum(beta**2 * precision + recall, 1e-12)
        best_threshold = thresholds[np.argmax(f_beta[:-1])]#last point has no threshold, so it is left out
        if np.isfinite(best_threshold) and best_threshold > 0:#safety: keep the old 10% if no usable cutoff exists
            self.threshold = float(best_threshold)
        return self.threshold

    def find_optimal_threshold(self):
        #scores every TRAINING company by the share of class 1 among its k nearest OTHER training companies
        #(leave-one-out, so a company never counts as its own neighbour), then picks the cutoff that maximises
        #true positive rate - false positive rate (Youden's J = balanced accuracy) on the ROC curve.
        X = np.asarray(self.X_train, dtype=float)
        y = np.asarray(self.y_train).astype(int)
        squared_norms = np.sum(X**2, axis=1)
        class_1_proportions = np.empty(len(X))
        for start in range(0, len(X), 500):#chunks keep memory small; numpy does all the distances at once
            chunk = X[start:start+500]
            squared_distance = np.sum(chunk**2, axis=1)[:, None] + squared_norms[None, :] - 2 * chunk @ X.T
            squared_distance[np.arange(len(chunk)), np.arange(start, start+len(chunk))] = np.inf
            k_indices = np.argpartition(squared_distance, self.k - 1, axis=1)[:, :self.k]
            class_1_proportions[start:start+len(chunk)] = y[k_indices].mean(axis=1)
        fpr, tpr, thresholds = roc_curve(y, class_1_proportions)
        best_threshold = thresholds[np.argmax(tpr - fpr)]
        if np.isfinite(best_threshold) and best_threshold > 0:#safety: keep the old 10% if no usable cutoff exists
            self.threshold = float(best_threshold)
        return self.threshold

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y).astype(int)
        X_all = self._with_knn_feature(self.X_train, leave_one_out=True)
        #optimal threshold: Youden's J on out-of-fold probabilities, so it is chosen on training data only
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        out_of_fold = cross_val_predict(clone(self.booster), X_all, self.y_train, cv=cv, method="predict_proba")[:, 1]
        fpr, tpr, thresholds = roc_curve(self.y_train, out_of_fold)
        ######################################################################
        #best_threshold = self.find_optimal_threshold()
        best_threshold = self.find_optimal_threshold_beta()
        #best_threshold = 0.10
        ######################################################################
        if np.isfinite(best_threshold):
            self.threshold = float(best_threshold)
        self.booster.fit(X_all, self.y_train)

    def predict_proba(self, X_test):
        return self.booster.predict_proba(self._with_knn_feature(X_test))[:, 1]

    def predict(self, X_test):
        return np.array(self.predict_proba(X_test) >= self.threshold).astype(bool)

    def _predict(self, x_test):
        return int(self.predict_proba(np.asarray(x_test, dtype=float).reshape(1, -1))[0] >= self.threshold)