# I decided to go with the breast cancer dataset
# https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
from sklearn.datasets import load_breast_cancer

# load libraries
import numpy as np
from math import sqrt
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


from KNN_from_scratch import KNN

# check on running time
start_time = time.time()

# load dataset
df = load_breast_cancer()

for ii in df:
    # print keys from dataset
    print(f'* {ii}')

knn = KNN(target_classes=2)

dataset = knn.load_data(df['data'])
target = df['target']

df['data'].shape  # check the shape of the data
df['target'].shape  # compare the data shape to the target

# split the data to train, test
X_train, X_test, y_train, y_test = train_test_split(dataset, target, test_size = 0.30, random_state=6)

print(len(X_train), len(X_test), len(y_train), len(y_test))

# scale the data
X_train_scaled = knn.fit_transform(X_train)
X_test_scaled = knn.fit_transform(X_test)

# Fitting the data
knn.fit_(X_train_scaled, y_train)

# predicting on data
predictions = knn.predict(X_test_scaled)

# Calculating Accuracy
print(knn.accuracy(y_test, predictions))

# print classification report
print(classification_report(y_test, predictions, target_names=df['target_names']))

# compare some of the results
print('Original value: %d, Predicted value: %d.' % (y_test[10], predictions[10]))

# plot confusion matrix
cm = confusion_matrix(y_test, predictions) #, normalize='all')  # can normalize the output
cmd = ConfusionMatrixDisplay(cm, display_labels=['malignant', 'benign'])
cmd.plot()


end_time = time.time()
print(f'running time: {end_time - start_time} seconds')
