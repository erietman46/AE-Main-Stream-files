# external support
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# internal support
from preprocess import scale, inverse_scale


def train(X_train, y_train, degree):

    """
    TODO:
    Part 2, Step 2(a):
        - Build a polynomial regression model, train it and return it
    """


    # and return it
    return model, poly_features


def predict(X, model, poly_features, X_mean, X_std, y_mean, y_std):
    # standardize input
    X = scale(X, X_mean, X_std)

    """
    TODO:
    Part 2, Step 2(b):
        - Use the trained {model} to estimate outputs for given inputs {X} (don't forget to assemble the polynomial features)
    """



    # destandardize output
    y = inverse_scale(y, y_mean, y_std)

    # all done
    return y
