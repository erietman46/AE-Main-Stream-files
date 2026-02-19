#!/usr/bin/env python3

# support
import numpy as np

# internal support
from load_data import load_dataset
from plot import plot_scatter, plot_histograms
from preprocess import scale


if __name__ == '__main__':

    # load the data
    X, y = load_dataset()

    # 1(a) plot the distance and flight time data samples in a scatter plot
    plot_scatter(X, y, filename="scatter.png")

    # 1(b) plot the distance and flight time data samples in histograms
    plot_histograms(X, y, bins=10, filename="histograms.png")

    # 1(c) calculate mean value and standard deviation
    """
    TODO:
    Part 1, Step 1(c):
        - Edit the following code to calculate the mean value and standard deviation of
        distance and duration
    """
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    X_std = np.std(X)
    y_std = np.std(y)

    # report on the mean value and standard deviation
    print(f'Mean flight time: {y_mean:.2f} [min], mean distance: {X_mean:.2f} [km]')
    print(f'Std flight time: {y_std:.2f} [min], std distance: {X_std:.2f} [km]')

    # 2.a scale the dataset to zero mean and one standard deviation
    """
    TODO:
    Part 1, Step 2(a):
        - Apply function {scale} (implemented in file preprocess.py) to standardize the dataset
    """
    X_norm = scale(X, X_mean, X_std)
    y_norm = scale(y, y_mean, y_std)

    # 2.b verify that the scaled dataset has now zero mean and one standard deviation
    """
    TODO:
    Part 1, Step 2(b):
        - Edit the following code to verify that the mean and standard deviation of the standardized
        dataset are indeed equal to zero and one, respectively
    """
    X_norm_mean = np.mean(X_norm)
    y_norm_mean = np.mean(y_norm)
    X_norm_std = np.std(X_norm)
    y_norm_std = np.std(y_norm)
    print(f'Mean flight time: {y_norm_mean:.2f} [min], mean distance: {X_norm_mean:.2f} [km]')
    print(f'Std flight time: {y_norm_std:.2f} [min], std distance: {X_norm_std:.2f} [km]')

