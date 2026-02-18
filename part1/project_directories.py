import os

# main project directory
project_dir = os.path.dirname(__file__)

# data directory
dataset_dir = project_dir + '/data/'

# figures directory
plot_dir = project_dir + '/plot/'

# create plot directory if it does not exist
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)