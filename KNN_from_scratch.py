# load libraries
import numpy as np
from math import sqrt


# Build K-Nearest Neighbor classifier
class KNN:
    '''K-Nearest Neighbour Classifier'''
    def __init__(self, target_classes):
        '''Specify a number of target classes'''
        self.target_classes = target_classes

    def load_data(self, filename):
        '''loads array list'''
        dataset = list()
        for row in filename:
            dataset.append(row)
        return dataset

    # StandardScaler
    def fit(self, X):
        '''scale the data'''
        self.mean_X = np.mean(X, axis=0)
        self.scaled_X = np.std(X - self.mean_X, axis=0)
        return self

    def transform(self, X):
        '''transform data'''
        return (X - self.mean_X) / self.scaled_X

    def fit_transform(self, X):
        '''fit_transform data'''
        return self.fit(X).transform(X)

    def fit_(self, X, y_train):
        '''fit train data before predict'''
        self.X_train = X
        self.y_train = y_train

    def euclidean_distance(self, row1, row2):
        '''Euclidian distance'''
        return np.sqrt(np.sum((row1 - row2) ** 2))

    def predict(self, X_test):
        '''predict the distance of KNN'''
        y = np.zeros(len(X_test))

        # iterate through the test set
        for ii in range(len(X_test)):

            # distance between test indices and all of the training set indices
            distance = np.array([self.euclidean_distance(X_test[ii], x_ind) for x_ind in self.X_train])

            # sort index from ascending to descending order of target classes
            distance_sorted = distance.argsort()[:self.target_classes]

            # for each neighbor find the target_class
            nearest_label = [self.y_train[ii] for ii in distance_sorted]

            y[ii] = max(set(nearest_label), key=nearest_label.count)

        return y

    def accuracy(self, y_test, y_pred):
        '''print an accuracy score'''
        return f'Accuracy score: {np.mean(y_test==y_pred)}'
