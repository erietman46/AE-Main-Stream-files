# support
import numpy as np

# internal support
from project_directories import dataset_dir


def load_dataset():
    data = np.loadtxt(dataset_dir + "flight_duration.csv", delimiter=',')
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1]
    return X, y
