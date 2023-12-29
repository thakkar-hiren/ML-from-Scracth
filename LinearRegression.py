import numpy as np

class LinearRegression:
    def __init__(self, learning_rate = 0.0002, n_iterations = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        no_of_samples, no_of_features = X.shape
        self.weights = np.zeros(no_of_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            y_predict = np.dot(X, self.weights) + self.bias

            # Computing gradients
            dw = (1/no_of_samples) * np.dot(X.T, (y_predict - y))
            db = (1/no_of_samples) * np.sum(y_predict-y)
            
            # Updating weights
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db


    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred