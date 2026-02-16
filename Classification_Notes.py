from matplotlib import pyplot as plt
import scipy as sp
import pandas as pd
import numpy as np
from sklearn import svm #This is the support vector machine library that we will be using for classification
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# We will use the XORdataset for this example. It is a simple dataset that is not linearly separable, which makes it a good example for classification.

dataset = np.loadtxt('XORdataset.csv', delimiter=',', dtype=int, skiprows=1) # Load the dataset from a CSV file
# Difference between pandas.read_csv and numpy.loadtxt is that pandas.read_csv returns a DataFrame.
# np.loadtxt returns a numpy array. Delimiter=',' specifies that the values are separated by commas. skiprows=1 tells the function to skip the first row of the file.

lst = [",".join(str(x) for x in row) for row in dataset.tolist()] 
# This line of code is converting the numpy array 'dataset' into a list of strings. Each row of the dataset is converted into a string where the values
# are separated by commas. The resulting list 'lst' will contain strings that represent each row of the dataset.
# It returns a list of string, where each string is for example '0,0,0' or '1,1,1' depending on the values in the dataset. 
# This is useful for creating a histogram of the sample types.


fig,ax = plt.subplots() # Create a figure and axis for plotting
plt.hist(lst)
plt.ylabel('# of samples')
plt.xlabel('Sample type')
#Result is a histogram that shows the frequency of each sample type in the dataset. 
#The x-axis represents the different sample types (e.g., '0,0,0', '0,1,1', etc.), and the y-axis represents the number of samples for each type.

#In the dataset, the first two columns represent the features (X1 and X2), and the third column represents the target t, which is eather 0 or 1.

X=dataset[:,0:2] # First is : , which means we want all the rows. Then 0:2 means we want the first two columns (X1 and X2) as our features.
t=dataset[:,2] # This line of code is selecting the third column (index 2) of the dataset, which contains the target variable 't'.


#CREATING THE SVM Classifier

clf = svm.SVC(kernel='linear') # Ok, lets break it down. 
# SVC stands for Support Vector Classifier, which is a type of SVM used for classification, not regression
# kernel specifices the type of decision boundary we want to create. 'linear' means we want a linear decision boundary.
# in 2D, it is a line. in 3D, it is a plane. In higher dimensions, it is a hyperplane.
# We can also use a gaussian kernel (RBF) for non-linear decision boundaries. RBF is a popular choice for non-linear classification problems. 
# clf is the model we will train on our dataset. It is an instance of the SVC class with a linear kernel.

clf.fit(X,t) # This line of code is training the SVM classifier on the dataset.
# What it does it takes the features X and the target t and finds the optimal hyperplane that separates the classes in the feature space.
# It uses the optimization algorithm to find the hyperplane that maximizes the margin between the classes, like we saw in the SVM lecture.


# Okay, so let's try the model
samples = [[0,0], [0,1], [1,0], [1,1]] # These are the four possible combinations of the two features (X1 and X2).
clf.predict(samples) # This line of code is using the trained SVM classifier to make predictions on the new samples, or unseen data. 
# It returns an array of predicted targets for each sample.  

#However, something went wrong! The sample [0,1] returns a target of 0. The problem is that kernel='linear' assumes that the data
# is lienarly separable, which is not the case for the XOR dataset. in this case, there is no decision boundary that can separate the classes perfectly.
# This is where kernels come in to help us.




# SVM WITH KERNELS


# What kernels do:
# Kernels compute the similarity between TWO SAMPLES (feature vectors).
# They allow the SVM to implicitly work in a higher dimensional space
# where the data may become linearly separable.
#
# IMPORTANT:
# X_1 and X_2 are rows from the dataset (two samples),
# NOT columns representing individual features.


# 1. Linear kernel
# K(X_1, X_2) = X_1 * X_2
# where X_1 and X_2 are two samples.
# The operation is the dot product between the vectors.
# The resulting decision boundary is linear (a line, plane, or hyperplane).
# Works well when the data is already approximately linearly separable.


# 2. Polynomial kernel
# K(X_1, X_2) = (X_1 * X_2 + 1)^d
# where d is the degree of the polynomial (a hyperparameter).
# This allows the classifier to create curved, more complex decision boundaries.
# Larger d -> more flexible model -> higher risk of overfitting.


# 3. RBF (Gaussian) kernel
# K(X_1, X_2) = exp(-gamma * ||X_1 - X_2||^2)
# where gamma is a hyperparameter controlling how fast similarity
# decreases as the distance between samples increases.
#
# Large gamma  -> narrow influence -> can overfit.
# Small gamma  -> wide influence   -> can underfit.
#
# Very popular choice when the true boundary shape is unknown,
# because it can model very complex relationships.


# 4. Sigmoid kernel
# K(X_1, X_2) = tanh(gamma * X_1 * X_2 + coef0)
# where gamma and coef0 are hyperparameters.
# It behaves somewhat like an activation function in neural networks.
# Sometimes faster, but often less reliable than RBF for difficult datasets.

clf=svm.SVC(kernel='rbf', gamma=1) # We are using the RBF kernel with a gamma of 1.

clf.fit(X,t) # Train the SVM with the RBF kernel on the dataset.

# Let's plot the decision boundary to see how it looks.
from matplotlib.colors import ListedColormap
fig, ax = plt.subplots()
ax.set_aspect('equal', 'box')

# Settings for the plot
timestep = 0.01
margin = 0.2

# Plotting the decision boundary, we use meshgrid to create a grid of points to evaluate the classifier
X1, X2 = np.meshgrid(np.arange(start = 0-margin, stop = 1 + timestep+margin, step = timestep),
                     np.arange(start = 0-margin, stop = 1 + timestep+margin, step = timestep))
result = clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)

plt.contourf(X1, X2, result,
             alpha = 0.75, cmap = ListedColormap(['r', 'g']))

# Plotting the points the function is defined on
X_set, y_set = np.array([[0,0], [0,1], [1,0], [1,1]]), np.array([0, 1, 1, 0])
plt.scatter(X_set[:, 0], X_set[:, 1], c = y_set, cmap = ListedColormap(['r', 'g']), marker='o', edgecolors='black')

# Plotting the axes
plt.title('Classifier on XOR dataset')
plt.xlabel('x1')
plt.ylabel('x2')

plt.show()

# Now we will see what kernel is best for the XOR dataset by comparing the accuracy of different kernels. We will use classification_report 
# from sklearn to evaluate the performance of the classifiers.

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for kernel in kernels:
    clf = svm.SVC(kernel=kernel, gamma=1) # Create an SVM classifier with the specified kernel and gamma
    clf.fit(X,t) # Train the classifier on the dataset
    predictions = clf.predict(X) # Make predictions on the training data
    print(f"Kernel: {kernel}")
    print(metrics.classification_report(t, predictions)) # Print the classification report for the predictions

# note that usually we should use test_split_train to get a training set and a test set, and we evaluate the performance of the classifier on the test set and train
# on the training set. 





#DATA NORMALIZATION USING STANDARD SCALER

dataset = pd.read_csv('customer_satisfaction.csv').dropna() # Load the dataset from a CSV file using pandas, dropna is used to drop any rows with missing values

X=dataset.drop('satisfaction', axis=1) # This line of code is selecting all the columns of the dataset except for the 'satisfaction' column, which is our target variable.
t=dataset['satisfaction'] # This line of code is selecting the 'satisfaction' column of the dataset, which contains the target variable 't'.



scaler = StandardScaler()  # what this does is it standardizes the features by removing the mean and scaling to unit variance.
# So for example X=40 becomes (40 - mean) / std, so just basically the equivalent Z-value like with a normal distribution. 
# This way the data values are very close to each other in regards to other features. This is important because datasets with high 
# variance compared to other features can dominate the decision boundary, which can lead to bad performance of the SVM.

#This is important for SVMs because they are sensitive to the scale of the features.
scaler.fit(X) 
scaled_X = scaler.transform(X)


