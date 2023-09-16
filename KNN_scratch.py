import numpy as np
from collections import Counter

def euclidian_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    def __init__(self, k=3) :
        self.k = k

    def fit(self,X,y):
        '''
        Function for training the data
        '''
        self.X_train = X
        self.y_train = y

    def predict(self,X):
        '''
        Function for predicting label
        '''
        preds = [self.predict_single_data_point(x) for x in X]
        return preds

    def predict_single_data_point(self,x):
        distance = [euclidian_distance(x,x_train) for x_train in self.X_train ]
        indices = np.argsort(distance)[:self.k]
        labels = [self.y_train[i] for i in indices]
        most_common_label = Counter(labels).most_common()
        return most_common_label[0][0]
