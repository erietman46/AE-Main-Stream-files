# support
from sklearn.model_selection import train_test_split
import numpy as np


def dataset_split(X, y, fraction):
    # split data into two subsets {a} and {b} so that:
    #   - {X_a} and {y_a} contain a fraction of the dataset equal to {fraction}
    #   - {X_b} and {y_b} contain a fraction of the dataset equal to {1-fraction}
    X_a, X_b, y_a, y_b = train_test_split(X, y, test_size=1.0-fraction, random_state=0, shuffle=True)

    # return split datasets
    return X_a, y_a, X_b, y_b


def scale(X, a, b):
    # return the scaled data
    return (X - a) / b


def inverse_scale(X, a, b):
    # return the scaled data
    return b * X + a


def standardize(X_train, y_train, X_test, y_test):
    """
    TODO:
    Part 1, Step 2.2:
        - Complete the following code to standardize the dataset
    """

    # get the mean and standard deviation on the training set
    X_mean = 0.0
    X_std = 0.0
    y_mean = 0.0
    y_std = 0.0

    # standardize all data
    # ...

    # return the standardized data and the mean and standard deviation of the training data
    return X_train, y_train, X_test, y_test, X_mean, y_mean, X_std, y_std