from matplotlib import pyplot as plt
import scipy as sp
import pandas as pd
import numpy as np
from sklearn import svm #This is the support vector machine library that we will be using for classification
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Exploring the Dataset

dataset = pd.read_csv('customer_satisfaction.csv').dropna() # Load the dataset from a CSV file using pandas, dropna is used to drop any rows with missing values

X=dataset.drop('satisfaction', axis=1) # This line of code is selecting all the columns of the dataset except for the 'satisfaction' column, which is our target variable.
t=dataset['satisfaction'] # This line of code is selecting the 'satisfaction' column of the dataset, which contains the target variable 't'.



scaler = StandardScaler()  # what this does is it standardizes the features by removing the mean and scaling to unit variance.
#This is important for SVMs because they are sensitive to the scale of the features. Still needs to be applied to the data. 
scaler.fit(X)  # This line of code is fitting the StandardScaler to the data in X,
# It calculaes the mean and standard deviation of each feature in X, which will be used for scaling the data.
X_scaled = scaler.transform(X) # Now we have the scaled data in X_scaled, which is the result of applying the transformation to X.
# Returns of the same panda series but with the values scaled.



# Obtain the training, test and validation sets

def train_test_validation_split(X, y, test_size, cv_size): # Function to split the data into training, testing, and cross-validation sets, done with random shuffling 
    #and a specified random state for reproducibility
    
    # collective size of test and cv sets, so test size and validation size together
    test_cv_size = test_size+cv_size 

    # split data into train and test - cross validation subsets
    X_train, X_testcv, y_train, y_testcv = train_test_split(
        X, y, test_size=test_cv_size, random_state=0, shuffle=True)

    # split test - cross validation sets into test and cross validation subsets
    X_test, X_cv, y_test, y_cv = train_test_split(
        X_testcv, y_testcv, test_size=cv_size/test_cv_size, random_state=0, shuffle=True)

    # return split data
    return [X_train, y_train, X_test, y_test, X_cv, y_cv]

X_train, t_train, X_test, t_test, X_cv, t_cv = train_test_validation_split(X_scaled, t, 0.2, 0.1).to_numpy() 

# Classification with SVMs

kernel_list = ['linear', 'poly', 'rbf', 'sigmoid'] # List of kernels to evaluate

false_positives_list = []

for kernel in kernel_list:
    clf = svm.SVC(kernel = kernel, gamma=1)
    clf.fit(X_train, t_train) # Train the classifier on the training data, returns a 
    t_predict_cv = clf.predict(X_cv) # Make predictions on the test data
    









