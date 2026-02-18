#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# internal support
from load_data import load_dataset
from preprocess import dataset_split, standardize
from regression import train
from project_directories import plot_dir


if __name__ == '__main__':

    # load the data
    X, y = load_dataset()

    # split the data into training-testing 70-30 parts by using a random function
    X_train, y_train, X_test, y_test = dataset_split(X, y, fraction=0.7)

    # 5. Train the linear regression model using only a portion of the training data from 10% to
    # 100% in increments of 10%
    # create lists to store the percentages, intercepts, coefficients and mses
    percentages = np.arange(0.1, 1.1, 0.1)
    intercepts = np.zeros(len(percentages))
    coeffs = np.zeros(len(percentages))
    mses = np.zeros(len(percentages))

    # train models for each percentage
    for percentage in percentages:
        """
        TODO:
        Part 1, Step 2.5:
            - Train the linear regression model using only a {percentage} of the training data
            - Assess the performance of the trained models by computing the Mean Square Error (MSE)
            on the test set
            - Store α, β and the MSE in {intercepts}, {coeffs}, {mses}, respectively. They will be
            plotted after this for loop
        """

    # plot the α and β values as a function of the percentage of the training data used
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(percentages*100, intercepts, linestyle='-', marker='o')
    ax[0].set_title('alpha')
    ax[0].set_xlabel('Percentage [%]')
    ax[0].set_ylabel('Intercept')
    ax[1].plot(percentages*100, coeffs, linestyle='-', marker='o')
    ax[1].set_title('beta')
    ax[1].set_xlabel('Percentage [%]')
    ax[1].set_ylabel('Coefficient')
    plt.savefig(plot_dir + "alpha_beta.png")

    # plot the mean squared error as a function of the percentage of the training data used
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(percentages*100, mses, linestyle='-', marker='o')
    ax.set_title('Mean Squared Error')
    ax.set_xlabel('Percentage [%]')
    ax.set_ylabel('MSE')
    ax.grid()
    plt.savefig(plot_dir + "mse.png")
