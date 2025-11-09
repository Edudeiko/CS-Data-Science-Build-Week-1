"""K-Nearest Neighbors classifier implementation from scratch."""
import numpy as np
from typing import Optional


class KNN:
    """K-Nearest Neighbors Classifier.

    Parameters:
        k (int): Number of nearest neighbors to consider for classification.
    """

    def __init__(self, k: int = 5):
        """Initialize KNN classifier with k neighbors."""
        self.k = k
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNN':
        """Store training data.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,).

        Returns:
            self: The fitted classifier.
        """
        self.X_train = X
        self.y_train = y
        return self

    def euclidean_distance(self, row1: np.ndarray, row2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points.

        Args:
            row1: First data point.
            row2: Second data point.

        Returns:
            float: Euclidean distance.
        """
        return np.sqrt(np.sum((row1 - row2) ** 2))

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class labels for test data.

        Args:
            X_test: Test features of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted labels of shape (n_samples,).
        """
        y_pred = np.zeros(len(X_test))

        for i in range(len(X_test)):
            # Calculate distances to all training points
            distances = np.array([
                self.euclidean_distance(X_test[i], x_train)
                for x_train in self.X_train
            ])

            # Get indices of k nearest neighbors
            k_nearest_indices = distances.argsort()[:self.k]

            # Get labels of k nearest neighbors
            k_nearest_labels = [self.y_train[idx] for idx in k_nearest_indices]

            # Vote: most common label wins
            y_pred[i] = max(set(k_nearest_labels), key=k_nearest_labels.count)

        return y_pred

    def accuracy(self, y_test: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy score.

        Args:
            y_test: True labels.
            y_pred: Predicted labels.

        Returns:
            float: Accuracy score between 0 and 1.
        """
        return np.mean(y_test == y_pred)


class StandardScaler:
    """Standardize features by removing mean and scaling to unit variance.

    The standard score of a sample x is calculated as:
        z = (x - mean) / std
    """

    def __init__(self):
        """Initialize StandardScaler."""
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """Compute the mean and std to be used for scaling.

        Args:
            X: Training data of shape (n_samples, n_features).

        Returns:
            self: Fitted scaler.
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features using previously computed mean and std.

        Args:
            X: Data to transform of shape (n_samples, n_features).

        Returns:
            np.ndarray: Transformed data.
        """
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it.

        Args:
            X: Data to fit and transform of shape (n_samples, n_features).

        Returns:
            np.ndarray: Transformed data.
        """
        return self.fit(X).transform(X)
