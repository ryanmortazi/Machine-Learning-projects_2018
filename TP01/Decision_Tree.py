# -*- coding: utf-8 -*-
"""
Created on Thu May 24 17:57:57 2018

@author: RyanM
"""

# In[ ]:


# Imports
import numpy as np
# to visualize the tree you must install this library
# conda install python-graphviz OR pip install graphviz
import graphviz
from sklearn import tree


# In[ ]:


# Load data from file
# File with 4 features extracted from Simpsons
num_features = 4
data_train = np.loadtxt("feature_vector_test.csv", delimiter=",")

# In[ ]:

# Define the training set
# Just for two classes (Smooth=0 and Spiral =1)

X_train  = data_train[:,0:num_features]
Y_train  = data_train[:,num_features]

# In[ ]:

# Train the Decision Tree with the training set
model = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=10)
model = model.fit(X_train, Y_train)


# In[ ]:
#two classes (Smooth=0 and Spiral =1)

dot_data = tree.export_graphviz(model, out_file=None, 
                         feature_names = ['red', 'blue', 'ratio_red2blue', 'ration_black2whitePixel'],  
                         class_names = ['Smooth', 'Spiral'],
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render("SpiralGalaxy_Data", view=True) 
graph 
# In[ ]:
# predict the class of samples
# train dataset
Y_train_pred = model.predict(X_train)
Y_train_pred


# In[ ]:
# predict the probability of each class
# train dataset

Y_train_pred_prob = model.predict_proba(X_train)
Y_train_pred_prob

# In[ ]:
# evaluation metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# In[ ]:


acc_galaxy_train = accuracy_score(Y_train, Y_train_pred )
print("Correct classification rate for train dataset = "+str(acc_galaxy_train*100)+"%")


# In[ ]:

from sklearn.metrics import classification_report

# In[ ]:


target_names = ['Smooth', 'Spiral']
print( classification_report(Y_train, Y_train_pred, target_names=target_names))

cm_galaxy_train = confusion_matrix(Y_train, Y_train_pred )
cm_galaxy_train

# In[ ]:

import itertools
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


np.set_printoptions(precision=2)

# In[ ]:


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm_galaxy_train, classes= ['Smooth', 'Spiral'],
                      title='Confusion matrix, without normalization')

# In[ ]:


# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm_galaxy_train, classes= ['Smooth', 'Spiral'], normalize=True,
                      title='Confusion matrix, with normalization')


plt.show()
