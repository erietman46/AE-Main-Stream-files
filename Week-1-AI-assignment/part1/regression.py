# external support
from sklearn.linear_model import LinearRegression

# internal support
from preprocess import scale, inverse_scale
import numpy as np


def train(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def predict(distance, model, X_mean, X_std, y_mean, y_std):
    """
    TODO:
    Part 1, Step 2.4:
        - Use the trained {model} to estimate the flight duration for a given {distance}
    """

    # convert distance to numpy array with correct shape
    distance = np.array(distance).reshape(-1, 1)

    # standardize input distance
    distance_norm = scale(distance, X_mean, X_std)

    # predict standardized time
    time_norm = model.predict(distance_norm)

    # convert back to original scale
    time = inverse_scale(time_norm, y_mean, y_std)

    # all done
    return np.array(time).flatten()