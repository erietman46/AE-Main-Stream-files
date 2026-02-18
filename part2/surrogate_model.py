#!/usr/bin/env python3

# external support
import numpy as np

# internal support
from aero_coefficients import run_aero as run_aero_expensive
from polyregression import train, predict
from preprocess import standardize
from project_directories import plot_dir


class SurrogateModel:

    # initializer
    def __init__(self, degree, N_alpha, alpha_min, alpha_max):
        # the degree of the polynomial regression
        self.degree = degree
        # the number of angles to sample to build the dataset grid
        self.N_alpha = N_alpha
        # the minimum alpha to sample to build the dataset grid
        self.alpha_min = alpha_min
        # the maximum alpha to sample to build the dataset grid
        self.alpha_max = alpha_max
        # the underlying regression model
        self.model = None
        # the polynomial features
        self.poly_features = None
        # the mean of the features
        self.X_mean = None
        # the standard deviation of the features
        self.X_std = None
        # the mean of the output
        self.Y_mean = None
        # the standard deviation of the output
        self.Y_std = None

    def _build_training_dataset(self):
        """
        Sample the expensive aerodynamic model on a grid of alpha values.
        Returns:
            X : (N, 1) array [alpha_deg]
            Y : (N, 2) array [CL, CD]
        """
        alpha_vals = np.linspace(self.alpha_min, self.alpha_max, self.N_alpha)

        """
        TODO:
        Part 2, Step 3(a):
            - Build the training data by evaluating the expensive model (run_aero_expensive) on a grid of alpha values.
            - Store the input alpha values in X and the corresponding CL and CD values in Y.
        """
        X_list = []
        Y_list = []



        X = np.asarray(X_list, dtype=float)
        Y = np.asarray(Y_list, dtype=float)
        return X, Y

    def train(self):
        """
        Train the polynomial regression surrogate.
        """
        # build training data from expensive model
        X, Y = self._build_training_dataset()

        # standardize
        Xs, Ys, self.X_mean, self.X_std, self.Y_mean, self.Y_std = standardize(X, Y)

        # train polynomial regression model
        self.model, self.poly_features = train(Xs, Ys, degree=self.degree)

        # all done
        return

    def run_aero(self, alpha_deg: float):
        """
        Surrogate aerodynamic model.

        On first call, it trains the surrogate using the expensive model.
        Subsequent calls are fast polynomial regression predictions.
        """
        X = np.array([[alpha_deg]], dtype=float)
        Y = predict(X, self.model, self.poly_features, self.X_mean, self.X_std, self.Y_mean, self.Y_std)

        # ensure scalar outputs
        Y = np.asarray(Y, dtype=float)
        CL = float(Y[0, 0])
        CD = float(Y[0, 1])
        return CL, CD


def plot_aero_coefficients_with_surrogate(filename: str = "aero_coefficients_surrogate.png"):
    """
    Plot the lift and drag coefficients as a function of angle of attack,
    overlaying the surrogate model on top of the expensive model results.
    """
    import matplotlib.pyplot as plt

    # Define range for alpha (same as plot_aero_coefficients.py)
    alpha_range = np.linspace(-10, 20, 30)  # degrees

    # Expensive model (scatter)
    CL_exp = np.zeros_like(alpha_range)
    CD_exp = np.zeros_like(alpha_range)
    for i in range(alpha_range.size):
        CL_exp[i], CD_exp[i] = run_aero_expensive(alpha_range[i])

    """
    TODO:
    Part 2, Step 4(a):
        - Modify the surrogate configuration parameters below and observe qualitatively how that affects the performance of the surrogate model
    """
    # Surrogate model (line) on a denser grid for smoothness
    surrogate_model = SurrogateModel(degree = 4, N_alpha = 10, alpha_min = -6.0, alpha_max = 14.0)
    surrogate_model.train()
    alpha_line = np.linspace(alpha_range.min(), alpha_range.max(), 200)
    CL_surr = np.zeros_like(alpha_line)
    CD_surr = np.zeros_like(alpha_line)
    for i in range(alpha_line.size):
        CL_surr[i], CD_surr[i] = surrogate_model.run_aero(alpha_line[i])

    # Plotting
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121)
    ax1.scatter(alpha_range, CL_exp, label="Expensive model")
    ax1.plot(alpha_line, CL_surr, label="Surrogate", color="tab:orange")
    ax1.set_title('Lift Coefficient (CL)')
    ax1.set_xlabel('Angle of Attack (degrees)')
    ax1.set_ylabel('Coefficient')
    ax1.set_ylim(-1, 1.45)
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.scatter(alpha_range, CD_exp, label="Expensive model")
    ax2.plot(alpha_line, CD_surr, label="Surrogate", color="tab:orange")
    ax2.set_title('Drag Coefficient (CD)')
    ax2.set_xlabel('Angle of Attack (degrees)')
    ax2.set_ylabel('Coefficient')
    ax2.set_ylim(0.01, 0.12)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(plot_dir + filename)


if __name__ == "__main__":
    plot_aero_coefficients_with_surrogate()
