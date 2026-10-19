import numpy as np

class LinearRegression:
    def __init__(self, n_iters=1000, lr=0.001 ):
        self.n_iters=n_iters
        self.lr=lr
        self.weights=None
        self.bias=None
    
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64) ##data type conversion to ensure no type error.
        y = np.asarray(y, dtype=np.float64).flatten()
        n_samples, n_features = X.shape ##the number of samples and features corresspond to the (row,column) tuple returned by the .shape() numpy method 
        self.weights = np.zeros(n_features) ##this makes a zero filled numpy array for our weights
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias ##creates a numpy array with each sample prediction starting with all the weights as zero and then with the updated weights as we update them.

            dw = (1/n_samples) * np.dot(X.T, y_pred - y) ##partial differential of the cost function with respect to weight. Summation is done by np.dot() already.
            db = (1/n_samples) * np.sum(y_pred-y) ##partial differential with respect to bias

            self.weights = self.weights-(self.lr*dw) ##updates the weights
            self.bias = self.bias-(self.lr*db) ##updates the biases

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        
        y_pred = np.dot(X, self.weights) + self.bias
        y_pred_binary = (y_pred > 0.5).astype(int)
        return y_pred_binary



