#In this file, you can see all you need to know for basic classification methods in Python
#Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

#Start with the simple so-called eXclsusively OR (XOR) problem, which is a common example in classification problems.

#Say x1 and x2 are two features, that can take values 0 or 1. The XOR associates to each pair (x1, x2) a label t, which is 1 if exactly one of the two features is 1, and 0 otherwise. 

dataset = np.loadtxt("XORdataset.csv", delimiter=",", dtype = int, skiprows = 1) #prints the dataset of x1, x2 and t as a list of lists, where each inner list is a row of the dataset. The first row is skipped because it contains the header.

# This line splits the dataset into a 1D list of strings we can plot
lst = [",".join(str(x) for x in row) for row in dataset.tolist()]

# Here we plot the data
fig, ax = plt.subplots()
plt.hist(lst)
plt.ylabel('# of samples')
plt.xlabel('Sample Type')
plt.show()

#dataset([2]) means the second column of the dataset, skipping the first row. This appears as [x1 x2 t]=[1 1 0]
#In order to train the model, we need to split the dataset the 2D feature array X and the target t.
X = dataset[:,0:2] #: means all rows, 0:2 means the first two columns, which are x1 and x2
t = dataset[:,2] #: means all rows, 2 means the third column, which is t

#Build a linear support vector classifier (SVC) model and fit it to the data
clf = svm.SVC(kernel='linear') #kernel='linear' means we are using a linear SVM
# train the support vector classifier
clf.fit(X, t)

# Setting up the plot settings
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

#If you run the code, you can see that the decision boundary is a straight line, which cannot separate the two classes of points (0,1) and (1,0) from the other two points (0,0) and (1,1). This is because the XOR problem is not linearly separable.

#A kernal function is a function that computes the similarity between two pairs of data points in a high dimensional space. In this higher-dimensional feature space, the dataset is linearly separable, and the SVM can find a hyperplane that separates the two classes. The kernel function allows us to compute the inner product of the data points in this high-dimensional space without explicitly mapping them to that space, which is computationally efficient.
#Types of Kernels include:
#1. Linear Kernel: K(X1,X2) = X1 \cdot X2 --> linearly separable data
#2. Polynomial Kernel: K(X1,X2) = (X1 \cdot X2 + c)^d --> non-linearly separable data, where c is a constant and d is the degree of the polynomial
#3. Sigmoid Kernel: K(X1,X2) = tanh(aX1 \cdot X2 + c) --> non-linearly separable data, where a and c are constant
#4. Radial Basis Function (RBF) Kernel: K(X1,X2) = exp(-gamma ||X1 - X2||^2) --> non-linearly separable data, where gamma is a constant that controls the width of the Gaussian function. This is the most used Kernel for nonlinear descision boundary.
#clf = svm.SVC(kernel='linear')
#clf = svm.SVC(kernel='poly', degree=2, coef0=1)
#clf = svm.SVC(kernel='sigmoid', coef0=0)
#clf = svm.SVC(kernel='rbf', gamma=1)






#EXAMPLE 2: IRIS DATASET
#The iris dataset is a classic dataset in machine learning, which contains measurements of 150 iris flowers from three different species: setosa, versicolor, and virginica. Each flower is described by four features: sepal length, sepal width, petal length, and petal width. The goal of the classification task is to predict the species of an iris flower based on these features.

iris = datasets.load_iris() #loads the iris dataset from sklearn
iris_names = iris.target_names #prints the names of the three species of iris flowers: ['setosa' 'versicolor' 'virginica']

#print(iris.data) prints the feature data of the iris dataset, which is a 150x4 array where each row corresponds to a flower and each column corresponds to a feature
#Data normalization is a common preprocessing step in machine learning, which involves scaling the features of the dataset to a common range. This can help improve the performance of many machine learning algorithms, including SVMs, by ensuring that all features contribute equally to the distance calculations used in the algorithm.

scaler = StandardScaler() #creates an instance of the StandardScaler class, which is used to standardize the features by removing the mean and scaling to unit variance
scaler.fit(iris.data) #fits the scaler to the data, which computes the mean and standard deviation for each feature
# We can find the mean and standard deviation of the dataset:
print(f'Mean: {scaler.mean_}')
print(f'Standard deviations: {scaler.scale_}')

X = scaler.transform(iris.data) #transforms the data using the fitted scaler, which standardizes the features by removing the mean and scaling to unit variance. The resulting array X has the same shape as the original data, but with standardized values for each feature.
t = iris.target #assigns the target labels of the iris dataset to the variable t, which is a 1D array of integers corresponding to the three species of iris flowers.


X_train, X_test, t_train, t_test = train_test_split(X, t, test_size = 0.2, random_state=0) #splits the dataset into training and testing sets, where X_train and t_train are the training data and labels, and X_test and t_test are the testing data and labels. The test_size parameter specifies the proportion of the dataset to include in the test split (20% in this case), and random_state is a seed for the random number generator to ensure reproducibility.

clf = svm.SVC(kernel='rbf') #creates an instance of the SVC class with the RBF kernel and a gamma value of 1
clf.fit(X_train, t_train) #fits the SVM model to the training data, which involves finding the optimal hyperplane that separates the classes in the feature space defined by X_train and t_train.

#USING THE TRAINED MODEL TO PREDICT THE CLASS OF A NEW SAMPLE
sepal_length = 5.1
sepal_width = 3.5
petal_length = 3.2
petal_width = 0.2
x = [sepal_length, sepal_width, petal_length, petal_width]
x_std = scaler.transform([x])
target = clf.predict(x_std)
iris.target_names[target[0]]

#ASSESSING THE PERFORMANCE OF THE MODEL
# predict the Iris types for the samples in the test set
predicted = clf.predict(X_test)

# assess performance of your classifier, where t_test are the true labels and predicted are the predicted labels by the classifier
print(f"Classification report for classifier {clf}:\n" f"{metrics.classification_report(t_test, predicted)}\n")




#EXAMPLE 3: CUSTOMER SATISFACTION DATASET
customerdataset = pd.read_csv("customer_satisfaction.csv") #Reads the customer satisfaction dataset
customerdataset = customerdataset.dropna() #Drops any rows with missing values

X_cust = customerdataset.drop('satisfaction', axis=1) #Creates the features vector by dropping the 'satisfaction' column
t_cust = customerdataset['satisfaction'] #Creates the target vector by selecting the 'satisfaction' column

#Scaling
scaler = StandardScaler()
scaler.fit(X_cust)
scaled_X_cust = scaler.transform(X_cust)

#Train, test and validation split
def train_test_validation_split(X, y, test_size, cv_size):
    # collective size of test and cv sets
    test_cv_size = test_size+cv_size

    # split data into train and test - cross validation subsets
    X_train, X_testcv, y_train, y_testcv = train_test_split(
        X, y, test_size=test_cv_size, random_state=0, shuffle=True)

    # split test - cross validation sets into test and cross validation subsets
    X_test, X_cv, y_test, y_cv = train_test_split(
        X_testcv, y_testcv, test_size=cv_size/test_cv_size, random_state=0, shuffle=True)

    # return split data
    return [X_train, y_train, X_test, y_test, X_cv, y_cv]

X_train, t_train, X_test, t_test, X_cv, t_cv = train_test_validation_split(scaled_X_cust, t_cust, 0.2, 0.1)

#List of Kernels to iterate over
kernel_list = ['linear', 'poly', 'sigmoid', 'rbf']
false_positives_list = [] #the number of false positives for each kernel will be stored in this list, and it needs to be as small as possible for a good classifier.

#Iteration
for kernel in kernel_list:
    clf = svm.SVC(kernel=kernel)
    clf.fit(X_train, t_train)

    t_pred_cv = clf.predict(X_cv)
    false_positives = (t_pred_cv != t_cv).sum()
    false_positives_list.append(false_positives)

print(false_positives_list)

#It turns out that the RBF kernel has the lowest number of false positives, so we will use that kernel to train our final model on the training set and evaluate it on the test set.
kernel = 'rbf'
clf = svm.SVC(kernel=kernel)
clf.fit(X_train, t_train)
