"""Test KNN implementation with breast cancer dataset.

Dataset: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import numpy as np
import time

from KNN_from_scratch import KNN, StandardScaler

# Check on running time
start_time = time.time()

# Load dataset
df = load_breast_cancer()

for key in df:
    # Print keys from dataset
    print(f'* {key}')

# Initialize KNN with k=5 neighbors (not number of classes)
knn = KNN(k=5)

# Get data and target
dataset = df['data']
target = df['target']

print(f"Data shape: {df['data'].shape}")
print(f"Target shape: {df['target'].shape}")

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    dataset, target, test_size=0.30, random_state=6
)

print(f"Train/Test split: {len(X_train)} / {len(X_test)}")

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit the KNN model
knn.fit(X_train_scaled, y_train)

# Make predictions
predictions = knn.predict(X_test_scaled)

# Calculate and print accuracy
accuracy = knn.accuracy(y_test, predictions)
print(f'Accuracy score: {accuracy:.4f}')

# Print classification report
print(classification_report(y_test, predictions, target_names=df['target_names']))

# Compare some of the results
print('Original value: %d, Predicted value: %d.' % (y_test[10], predictions[10]))

# Plot confusion matrix
cm = confusion_matrix(y_test, predictions)
cmd = ConfusionMatrixDisplay(cm, display_labels=['malignant', 'benign'])
cmd.plot()

end_time = time.time()
print(f'Running time: {end_time - start_time:.2f} seconds')
