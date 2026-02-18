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


def standardize(X, Y):
    """
    TODO:
    Part 2, Step 3(b):
        - Complete the following code to standardize the dataset
    """


    # return the standardized data and the mean and standard deviation of the training data
    return X_standardized, Y_standardized, X_mean, X_std, Y_mean, Y_std
