#!/usr/bin/env python3

# support
import matplotlib.pyplot as plt

# internal support
from project_directories import plot_dir


def plot_scatter(X, y, filename):
    """
    Plot the dataset in a scatter plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    ax.scatter(X, y, alpha=0.6)
    ax.set_xlabel("Flight Distance")
    ax.set_ylabel("Flight Duration")
    ax.set_title("Flight Distance vs Duration")

    plt.tight_layout()
    plt.savefig(plot_dir + filename)
    plt.close(fig)

def plot_histograms(X, y, bins, filename):
    """
    TODO:
    Part 1, Step 1(b):
        - Complete the following code to plot histograms of flight distance and duration
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].hist(X, bins=bins)
    ax[0].set_xlabel("Flight Distance")
    ax[0].set_ylabel("Frequency")
    ax[0].set_title("Distance Distribution")

    ax[1].hist(y, bins=bins)
    ax[1].set_xlabel("Flight Duration")
    ax[1].set_ylabel("Frequency")
    ax[1].set_title("Duration Distribution")

    plt.tight_layout()
    plt.savefig(plot_dir + filename)
    plt.close(fig)