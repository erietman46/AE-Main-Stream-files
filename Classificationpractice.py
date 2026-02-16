#In this file, you can see all you need to know for basic classification methods in Python
#Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm

#We will be using Suppport Vector Machine (SVM) for classification

#Start with the simple so-called eXclsusively OR (XOR) problem, which is a common example in classification problems.
#Say x1 and x2 are two features, that can take values 0 or 1. The XOR associates to each pair (x1, x2) a label t, which is 1 if exactly one of the two features is 1, and 0 otherwise. =