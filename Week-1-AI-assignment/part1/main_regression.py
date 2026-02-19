#!/usr/bin/env python3

# support
from load_data import load_dataset
from preprocess import dataset_split, standardize
from regression import train, predict
import numpy as np

if __name__ == '__main__':

    # load the data
    X, y = load_dataset()

    # 1. Split the data into training and test subsets 70-30
    """
    TODO:
    Part 1, Step 2.1:
        - Apply function dataset split (implemented in file preprocess.py) to split the dataset
        into the training and and test subsets (70%-30% split)
    """
    X_train, y_train, X_test, y_test = dataset_split(X, y, 0.7)

    # 2. Standardize the dataset
    X_train, y_train, X_test, y_test, X_mean, y_mean, X_std, y_std = standardize(X_train, y_train, X_test, y_test)

    # 3. Train the linear regression model and obtain the intercept α and slope β
    model = train(X_train, y_train)
    print(f'Intercept: {model.intercept_}, Coeff: {model.coef_[0]}')

    # 4. Estimate the time of flight for the three destinations presented before.
    # distances (Source: google maps measure)
    distance_milano = 830
    distance_seattle = 7840
    distance_brussels = 157

    # predict
    time_milano, time_seattle, time_brussels = predict([[distance_milano], [distance_seattle], [distance_brussels]], model, X_mean, X_std, y_mean, y_std)

    # report
    print(f'Flight time from Amsterdam to Milano: {time_milano:.2f} [min]')
    print(f'Flight time from Amsterdam to Seattle: {time_seattle:.2f} [min]')
    print(f'Flight time from Amsterdam to Brussels: {time_brussels:.2f} [min]')

print(X_train)