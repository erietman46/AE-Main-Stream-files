#!/usr/bin/env python3

from aero_coefficients import run_aero
from project_directories import plot_dir

def plot_aero_coefficients(filename: str = "aero_coefficients.png"):
    """
    Plot the lift and drag coefficients as a function of angle of attack and Mach number.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Define range for alpha
    alpha_range = np.linspace(-10, 20, 30)  # degrees

    # Create meshgrid for plotting
    CL = np.zeros_like(alpha_range)
    CD = np.zeros_like(alpha_range)

    # Compute CL and CD for each alpha
    for i in range(alpha_range.size):
        """
        TODO:
        Part 2, Step 1(c):
        - Complete the following code to plot the run_aero() dataset in a scatter plot
        """


    # Plotting
    fig = plt.figure(figsize=(12, 6))

    # Left plot: CL vs alpha
    ax1 = fig.add_subplot(121)
    ax1.scatter(alpha_range, CL)
    ax1.set_title('Lift Coefficient (CL)')
    ax1.set_xlabel('Angle of Attack (degrees)')
    ax1.set_ylabel('Coefficient')

    # Right plot: CD vs alpha
    ax2 = fig.add_subplot(122)
    ax2.scatter(alpha_range, CD)
    ax2.set_title('Drag Coefficient (CD)')
    ax2.set_xlabel('Angle of Attack (degrees)')
    ax2.set_ylabel('Coefficient')

    plt.tight_layout()
    plt.savefig(plot_dir + filename)

if __name__ == "__main__":
    plot_aero_coefficients()
