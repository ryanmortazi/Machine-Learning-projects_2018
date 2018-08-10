# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 12:25:53 2018

@author: RyanM
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

num_features_spam = 57
num_features_galaxy = 75
data_spam = np.loadtxt("spam.csv", delimiter=",")
data_galaxy = np.loadtxt("galaxy_feature_vectors.csv", delimiter=",")

# Define the training set
X_data_spam  = data_spam[:,0:num_features_spam]
Y_data_spam  = data_spam[:,num_features_spam] # last column = class labels

X_data_galaxy = data_galaxy[:,0:num_features_galaxy]
Y_data_galaxy = data_galaxy[:,num_features_galaxy] # last column = class labels
# Using hold-out evaluation
from sklearn.model_selection import train_test_split
'''
 - Split the data into train and valid, holding 20% of the data into test 
 and 20% for valid
 - This is the split data for Holdout validation and later on use another data 
 split to evaluate the perfomance of k-fold CV.
 -
'''
X_train_spam, X_valid_spam, Y_train_spam, Y_valid_spam = train_test_split(
    X_data_spam, Y_data_spam, test_size=0.4, random_state=0, shuffle=True, stratify=Y_data_spam
)
X_test_spam, X_valid_spam, Y_test_spam, Y_valid_spam = train_test_split(
    X_valid_spam, Y_valid_spam, test_size=0.5, random_state=0, shuffle=True, stratify=Y_valid_spam
)
X_train_galaxy, X_valid_galaxy, Y_train_galaxy, Y_valid_galaxy = train_test_split(
    X_data_galaxy, Y_data_galaxy, test_size=0.4, random_state=0, shuffle=True, stratify=Y_data_galaxy
)
X_test_galaxy, X_valid_galaxy, Y_test_galaxy, Y_valid_galaxy = train_test_split(
    X_valid_galaxy, Y_valid_galaxy, test_size=0.5, random_state=0, shuffle=True, stratify=Y_valid_galaxy
)
# In[30]:

from sklearn.metrics import accuracy_score

# =============================================================================
# from sklearn.preprocessing import StandardScaler
# Xtrain_galaxy_s=X_train_galaxy
# Xtest_galaxy_s=X_test_galaxy
# Xvalid_galaxy_s=X_valid_galaxy
# =============================================================================
from mdlp.discretization import MDLP
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
mdlp = MDLP()
conv_X = mdlp.fit_transform(X, y)