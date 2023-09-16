import numpy as np
from KNN_scratch import KNN
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Loading the data...
dataset = load_iris()

# Getting Data and Target Variable
X,y = dataset.data, dataset.target

# Train-Test Split...
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating object of KNN
classifier = KNN(k=9)

# Training the classifier using fit method
classifier.fit(X_train, y_train)

# Predicting the labels for test data
predictions = classifier.predict(X_test)

# Computing accuracy of our classifier
accuracy = np.sum(predictions == y_test)/len(y_test)
print("Accuracy of the model:- ",accuracy)
