#!/usr/bin/env python3

# support
import matplotlib.pyplot as plt

# internal support
from project_directories import plot_dir


def plot_scatter(X, y, filename):
    """
    TODO:
    Part 1, Step 1(a):
        - Complete the following code to plot the dataset in a scatter plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    plt.savefig(plot_dir + filename)


def plot_histograms(X, y, bins, filename):
    """
    TODO:
    Part 1, Step 1(b):
        - Complete the following code to plot histograms of flight distance and duration
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plt.savefig(plot_dir + filename)