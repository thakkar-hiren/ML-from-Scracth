from LinearRegression import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

# Generating Data
X, y = datasets.make_regression(n_samples=200, n_features=1, noise=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def MSE(y_test, predictions):
    return np.mean((y_test - predictions)**2)

linear = LinearRegression(n_iterations=100000)
linear.fit(X_train, y_train)
predictions = linear.predict(X_test)
error = MSE(y_test, predictions)
print("MSE:- ",error)

pred_line = linear.predict(X)

fig = plt.figure(figsize=(8,8))
plt.scatter(X[:,0], y, color='g', marker='o')
plt.plot(X, pred_line, linewidth=2)
plt.show()