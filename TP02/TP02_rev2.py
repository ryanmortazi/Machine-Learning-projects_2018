
# coding: utf-8

'''
Pour l'algorithm de Bayes, les methodes MDLP et 
les la méthode non-supervisée sont des methodes de prétraitement. 
Pour le MDLP, vous pouvez installer le package en suivant les instructions 
de https://github.com/hlin117/mdlp-discretization
Pour la méthode de non-supervisé, vous pouvez faire le prétraitement MinMaxScaler 
(sklearn) à la place.
'''

# In[26]:


# Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from scipy.stats.mstats import mquantiles, kurtosis, skew
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split




# In[27]:


# Première étape: créer un ensemble de validation pour le dataset des pourriels

# Load data from file (converted from nominal to numerical (binary approach))
'''
Galaxy dataset lables: smooth=0, spiral=1
Emails dataset labels: spam (1) » ou « non-spam (0)

'''
num_features_spam = 57
num_features_galaxy = 75
data_spam = np.loadtxt("spam.csv", delimiter=",")
data_galaxy = np.loadtxt("galaxy_feature_vectors.csv", delimiter=",")


# In[30]:


# Define the training set
X_data_spam  = data_spam[:,0:num_features_spam]
Y_data_spam  = data_spam[:,num_features_spam] # last column = class labels

X_data_galaxy = data_galaxy[:,0:num_features_galaxy]
Y_data_galaxy = data_galaxy[:,num_features_galaxy] # last column = class labels


# In[36]:


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


# In[39]:

# Let's verify we have the statify option work
# We should have the same proportion of non-spam and spam labels inside both data sets
print(Y_train_spam.mean())
print(Y_valid_spam.mean())

print(Y_train_galaxy.mean())
print(Y_valid_galaxy.mean())
count_Ones=0
count_zeros=0
for i in Y_train_galaxy:
    if i==0:
        count_zeros+=1
    else:
        count_Ones+=1
print ("number of smooth galaxies in train dataset is {} and spiral galaxies {}".format(count_zeros,count_Ones))
count_Ones=0
count_zeros=0
for i in Y_test_galaxy:
    if i==0:
        count_zeros+=1
    else:
        count_Ones+=1
print ("number of smooth galaxies in test dataset is {} and spiral galaxies {}".format(count_zeros,count_Ones))

count_Ones=0
count_zeros=0
for i in Y_valid_galaxy:
    if i==0:
        count_zeros+=1
    else:
        count_Ones+=1
print ("number of smooth galaxies in validation dataset is {} and spiral galaxies {}".format(count_zeros,count_Ones))

count_Ones=0
count_zeros=0
for i in Y_train_spam:
    if i==0:
        count_zeros+=1
    else:
        count_Ones+=1
print ("number of non-spam emails in train dataset is {} and spam emails {}".format(count_zeros,count_Ones))
count_Ones=0
count_zeros=0
for i in Y_test_spam:
    if i==0:
        count_zeros+=1
    else:
        count_Ones+=1
print ("number of non-spam emails in test dataset is {} and spam emails {}".format(count_zeros,count_Ones))
count_Ones=0
count_zeros=0
for i in Y_valid_spam:
    if i==0:
        count_zeros+=1
    else:
        count_Ones+=1
print ("number of non_spam emails in valid dataset is {} and spam emails {}".format(count_zeros,count_Ones))
# In[60]:

# Produisez un code source permettant d’entraîner un modèle afin de classifier les pourriels avec l’aide des
# trois algorithmes vus en classe, soit :
# 1. un arbre de décision
# 2. Bayes naïf (nous allons devoir utiliser MultinomialNB)
# 3. KNN

'''
We are going to build Decision Tree models with different hyper-parameter combination and 
go through the whole training, validation process for every combination
'''
import graphviz
from sklearn import tree

feature_names = ['Couleur moyenne du centre [0]','Couleur moyenne du centre [1]',
 'Couleur moyenne du centre [2] ','Couleur moyenne [0]','Couleur moyenne [1]',
 'Couleur moyenne [2]','Standard Deviation [0]','Standard Deviation [1]',
 'Standard Deviation [2]','Distribution Kurtosis [0]','Distribution Kurtosis [1]',
 'Distribution Kurtosis [2]','Distribution normale asymétrique [0]','Distribution normale asymétrique [1]',
 'Distribution normale asymétrique [2]','Coefficient Gini [0]','Coefficient Gini [1]',
 'Coefficient Gini [2]','Excentricité','Largeur','Hauteur','Somme','Entropie',
 'Chiralité','Aire de l’ellipse','Aire box-to-image','Décalage du centre (offset)',
 'Rayon de la lumière [0] ','Rayon de la lumière [1] ','Nombre de labels',
 'Distribution Kurtosis [0] ','Distribution Kurtosis [1] ','Distribution Kurtosis [2] ',
 'Distribution normale asymétrique [0]','Distribution normale asymétrique [1] ',
 'Distribution normale asymétrique [2] ','Coefficient Gini [0]','Coefficient Gini [1]',
 'Coefficient Gini [2]','Distribution Kurtosis image noir et blanc',
 'Distribution normale asymétrique image noir et blanc','Coefficient Gini image noir et blanc',
 'Couleur du centre [0]','Couleur du centre [1]','Couleur du centre [2]',
 'Couleur moyenne [0]','Couleur moyenne [1]','Couleur moyenne [2]',
 'Couleur moyenne du centre [0]','Couleur moyenne du centre [1]','Couleur moyenne du centre [2]',
 'Couleur du centre ','Rapport couleur du centre / moyenne de gris','Moments de l’image',
 'Moments de l’image','Moments de l’image','Moments de l’image','Moments de l’image',
 'Moments de l’image','Moments de l’image','Moments de l’image','Moments de l’image',
 'Moments de l’image','Moments de l’image','Moments de l’image','Moments de l’image',
 'Moments de l’image','Moments de l’image','Moments de l’image','Moments de l’image',
 'Moments de l’image','Moments de l’image','Moments de l’image','Moments de l’image',
 'Moments de l’image']

# In[25]:


# Train the Decision Tree with the training set max_depth 10
model_depth10 = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=10, max_depth=10)
model_depth10 = model_depth10.fit(X_train_galaxy, Y_train_galaxy)

# Visualize the tree in jupyter and save it in a PNG file
dot_data = tree.export_graphviz(model_depth10, out_file=None, 
                         feature_names=feature_names,
                         class_names = ['Smooth', 'Spiral'],
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render("SpiralGalaxy_Data", view=True) 
graph 

# predict the class of samples
# validation dataset
Y_validation_pred_max_depth10 = model_depth10.predict(X_valid_galaxy)


# predict the probability of each class
# validation dataset

Y_validation_pred_prob_max_depth10 = model_depth10.predict_proba(X_valid_galaxy)


# In[26]:


# Train the Decision Tree with the training set max_depth 5
model_depth5 = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=10, max_depth=5)
model_depth5 = model_depth5.fit(X_train_galaxy, Y_train_galaxy)


dot_data = tree.export_graphviz(model_depth5, out_file=None, 
                         feature_names=feature_names,
                         class_names = ['Smooth', 'Spiral'],
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render("SpiralGalaxy_Data", view=True) 
graph 

# predict the class of samples
# validation dataset
Y_validation_pred_max_depth5 = model_depth5.predict(X_valid_galaxy)


# predict the probability of each class
# validation dataset

Y_validation_pred_prob_max_depth5 = model_depth5.predict_proba(X_valid_galaxy)

# In[27]:


# Train the Decision Tree with the training set max_depth 3
model_depth3 = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=10, max_depth=3)
model_depth3 = model_depth3.fit(X_train_galaxy, Y_train_galaxy)


dot_data = tree.export_graphviz(model_depth3, out_file=None, 
                         feature_names=feature_names,
                         class_names = ['Smooth', 'Spiral'],
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render("SpiralGalaxy_Data", view=True) 
graph 

# predict the class of samples
# validation dataset
Y_validation_pred_max_depth3 = model_depth3.predict(X_valid_galaxy)


# predict the probability of each class
# validation dataset

Y_validation_pred_prob_max_depth3 = model_depth3.predict_proba(X_valid_galaxy)

# In[27]:


# Train the Decision Tree with the training set max_depth None
model_depth_None = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=10)
model_depth_None = model_depth_None.fit(X_train_galaxy, Y_train_galaxy)


dot_data = tree.export_graphviz(model_depth_None, out_file=None, 
                         feature_names=feature_names,
                         class_names = ['Smooth', 'Spiral'],
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render("SpiralGalaxy_Data_None", view=True) 
graph 

# predict the class of samples
# validation dataset
Y_validation_pred_max_depth_None = model_depth_None.predict(X_valid_galaxy)


# predict the probability of each class
# validation dataset

Y_validation_pred_prob_max_depth_None = model_depth_None.predict_proba(X_valid_galaxy)


# In[32]:


# evaluation metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import itertools

# Method to plot confusion Matrix
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
# In[33]:


acc_galaxy_validation_max_depth10 = accuracy_score(Y_valid_galaxy, Y_validation_pred_max_depth10 )
print("Correct classification rate for validation dataset with max depth 10 = "+str(acc_galaxy_validation_max_depth10*100)+"%")

acc_galaxy_validation_max_depth5 = accuracy_score(Y_valid_galaxy, Y_validation_pred_max_depth5 )
print("Correct classification rate for validation dataset with max depth 5 = "+str(acc_galaxy_validation_max_depth5*100)+"%")

acc_galaxy_validation_max_depth3 = accuracy_score(Y_valid_galaxy, Y_validation_pred_max_depth3 )
print("Correct classification rate for validation dataset with max depth 3 = "+str(acc_galaxy_validation_max_depth3*100)+"%")

acc_galaxy_validation_max_depth_None = accuracy_score(Y_valid_galaxy, Y_validation_pred_max_depth_None )
print("Correct classification rate for validation dataset with max depth None = "+str(acc_galaxy_validation_max_depth_None*100)+"%")

# In[34]:

from sklearn.metrics import precision_recall_fscore_support


# In[35]:
# evaluating different hyperparameters for Decision tree models(i.e. Accuracy & F1_Score)
print("max_depth = 10")
precision10,_,fbeta_score10,_=precision_recall_fscore_support(Y_valid_galaxy,Y_validation_pred_max_depth10,average="weighted")
print ("precision is {} and F1-score is {}".format(precision10,fbeta_score10))
print("\n max_depth = 5")
precision5,_,fbeta_score5,_=precision_recall_fscore_support(Y_valid_galaxy,Y_validation_pred_max_depth5,average="weighted")
print ("precision is {} and F1-score is {}".format(precision5,fbeta_score5))

print("\n max_depth = 3")
precision3,_,fbeta_score3,_=precision_recall_fscore_support(Y_valid_galaxy,Y_validation_pred_max_depth3,average="weighted")
print ("precision is {} and F1-score is {}".format(precision3,fbeta_score3))

print("\n max_depth = None")
precision_None,_,fbeta_score_None,_=precision_recall_fscore_support(Y_valid_galaxy,Y_validation_pred_max_depth_None,average="weighted")
print ("precision is {} and F1-score is {}".format(precision_None,fbeta_score_None))

plt.figure("Accuracy")
x_precision = np.array([0,3,7,10])
y_precision = np.array([precision_None,precision3,precision5,precision10])
my_xticks = ['without max_depth','max_depth:3','max_depth:5','max_depth:10']
plt.xticks(x_precision, my_xticks)
plt.plot(x_precision, y_precision,'ro')

plt.figure("F1 Score")
x_F1Score = np.array([0,3,7,10])
y_F1Score = np.array([fbeta_score_None,fbeta_score3,fbeta_score5,fbeta_score10])
my_xticks = ['without max_depth','max_depth:3','max_depth:5','max_depth:10']
plt.xticks(x_F1Score, my_xticks)
plt.plot(x_F1Score, y_F1Score,'ro')
plt.show()

'''
Based on the accuracy and f1-score, the perfomance of model without max_depth 
is comparable to max_depth=10. Hence, we choose model without max_depth.
'''
# In[33]:
# Evaluating Test Dataset for galaxy with the chosen hyperparameter (i.e. Without max_depth)
Y_Test_pred_max_depth_None = model_depth_None.predict(X_test_galaxy)
Y_Test_pred_prob_max_depth_None = model_depth_None.predict_proba(X_test_galaxy)
acc_galaxy_test_max_depth_None = accuracy_score(Y_test_galaxy, Y_Test_pred_max_depth_None )
print("Correct classification rate for Test dataset with max depth None = "+str(acc_galaxy_test_max_depth_None*100)+"%")
cm_galaxy_test_max_depth_None = confusion_matrix(Y_test_galaxy, Y_Test_pred_max_depth_None )

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix for max depth =7
plt.figure()
plot_confusion_matrix(cm_galaxy_test_max_depth_None, classes= ['Smooth', 'Spiral'],
                      title='Confusion matrix, without normalization without max_depth')
plt.figure()
plot_confusion_matrix(cm_galaxy_test_max_depth_None, classes= ['Smooth', 'Spiral'], normalize=True,
                      title='Confusion matrix, with normalization without max_depth')
plt.show()
# In[33]:

# We are going to build K_NN models with different hyper-parameter combination and 
# go through the whole training, validation process for every combination
print("       K_NN models with hold-out set ")

# Scale data for train, validation (hold out), and test
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
Xtrain_galaxy_s = standard_scaler.fit_transform(X_train_galaxy)
Xtest_galaxy_s = standard_scaler.transform(X_test_galaxy)
Xvalid_galaxy_s = standard_scaler.transform(X_valid_galaxy)

# In[33]:
#Classify using k-NN for k=3,5,10 and weight= uniform
from sklearn.neighbors import KNeighborsClassifier
print("k=3 and weight= uniform")

knn_3u = KNeighborsClassifier(n_neighbors=3, weights="uniform")
knn_3u.fit(Xtrain_galaxy_s, Y_train_galaxy)

Y_validation_pred_3u = knn_3u.predict(X_valid_galaxy)
Y_validation_pred_prob_3u = knn_3u.predict_proba(X_valid_galaxy)

print("k=5 and weight= uniform")
knn_5u = KNeighborsClassifier(n_neighbors=5, weights="uniform")
knn_5u.fit(Xtrain_galaxy_s, Y_train_galaxy)

Y_validation_pred_5u = knn_5u.predict(X_valid_galaxy)
Y_validation_pred_prob_5u = knn_5u.predict_proba(X_valid_galaxy)

print("k=10 and weight= uniform")
knn_10u = KNeighborsClassifier(n_neighbors=10, weights="uniform")
knn_10u.fit(Xtrain_galaxy_s, Y_train_galaxy)

Y_validation_pred_10u = knn_10u.predict(X_valid_galaxy)
Y_validation_pred_prob_10u = knn_10u.predict_proba(X_valid_galaxy)

acc_galaxy_validation_3u = accuracy_score(Y_valid_galaxy, Y_validation_pred_3u )
print("Correct classification rate for validation dataset with k=3 and weight: uniform = "+str(acc_galaxy_validation_3u*100)+"%")
acc_galaxy_validation_5u = accuracy_score(Y_valid_galaxy, Y_validation_pred_5u )
print("Correct classification rate for validation dataset with k=5 and weight: uniform = "+str(acc_galaxy_validation_5u*100)+"%")
acc_galaxy_validation_10u = accuracy_score(Y_valid_galaxy, Y_validation_pred_10u )
print("Correct classification rate for validation dataset with k=10 and weight: uniform = "+str(acc_galaxy_validation_10u*100)+"%")


# In[33]:
#Classify using k-NN for k=3,5,10 and weight= distance
print("k=3 and weight= distance")
knn_3d = KNeighborsClassifier(n_neighbors=3, weights="distance")
knn_3d.fit(Xtrain_galaxy_s, Y_train_galaxy)

Y_validation_pred_3d = knn_3u.predict(X_valid_galaxy)
Y_validation_pred_prob_3d = knn_3d.predict_proba(X_valid_galaxy)

print("k=5 and weight= distance")
knn_5d = KNeighborsClassifier(n_neighbors=3, weights="distance")
knn_5d.fit(Xtrain_galaxy_s, Y_train_galaxy)

Y_validation_pred_5d = knn_5d.predict(X_valid_galaxy)
Y_validation_pred_prob_5d = knn_5d.predict_proba(X_valid_galaxy)

print("k=10 and weight= distance")
knn_10d = KNeighborsClassifier(n_neighbors=3, weights="distance")
knn_10d.fit(Xtrain_galaxy_s, Y_train_galaxy)

Y_validation_pred_10d = knn_10d.predict(X_valid_galaxy)
Y_validation_pred_prob_10d = knn_10d.predict_proba(X_valid_galaxy)

acc_galaxy_validation_3d = accuracy_score(Y_valid_galaxy, Y_validation_pred_3d )
print("Correct classification rate for validation dataset with k=3 and weight: distance = "+str(acc_galaxy_validation_3d*100)+"%")

acc_galaxy_validation_5d = accuracy_score(Y_valid_galaxy, Y_validation_pred_5d )
print("Correct classification rate for validation dataset with k=5 and weight: distance = "+str(acc_galaxy_validation_5d*100)+"%")

acc_galaxy_validation_10d = accuracy_score(Y_valid_galaxy, Y_validation_pred_10d )
print("Correct classification rate for validation dataset with k=10 and weight: distance = "+str(acc_galaxy_validation_10d*100)+"%")

# In[35]:
# evaluating different hyperparameters for Decision tree models(i.e. Accuracy & F1_Score)
print("k=10 and weight= uniform")
precision10u,_,fbeta_score10u,_=precision_recall_fscore_support(Y_valid_galaxy,Y_validation_pred_10u,average="weighted")
print ("precision is {} and F1-score is {}".format(precision10u,fbeta_score10u))
print("\n k=5 and weight= uniform")
precision5u,_,fbeta_score5u,_=precision_recall_fscore_support(Y_valid_galaxy,Y_validation_pred_5u,average="weighted")
print ("precision is {} and F1-score is {}".format(precision5u,fbeta_score5u))

print("\n k=3 and weight= uniform")
precision3u,_,fbeta_score3u,_=precision_recall_fscore_support(Y_valid_galaxy,Y_validation_pred_3u,average="weighted")
print ("precision is {} and F1-score is {}".format(precision3u,fbeta_score3u))

print("k=10 and weight= distance")
precision10d,_,fbeta_score10d,_=precision_recall_fscore_support(Y_valid_galaxy,Y_validation_pred_10d,average="weighted")
print ("precision is {} and F1-score is {}".format(precision10d,fbeta_score10d))
print("\n k=5 and weight= distance")
precision5d,_,fbeta_score5d,_=precision_recall_fscore_support(Y_valid_galaxy,Y_validation_pred_5d,average="weighted")
print ("precision is {} and F1-score is {}".format(precision5d,fbeta_score5d))

print("\n k=3 and weight= distance")
precision3d,_,fbeta_score3d,_=precision_recall_fscore_support(Y_valid_galaxy,Y_validation_pred_3d,average="weighted")
print ("precision is {} and F1-score is {}".format(precision3d,fbeta_score3d))

plt.figure("Accuracy")
x_precision = np.array([0,2,4,6,8,10])
y_precision = np.array([precision3u,precision5u,precision10u,precision3d,precision5d,precision10d])
my_xticks = ['k3_uni','k5_uni','k10_uni','k3_dis','k5_dis','k10_dis']
plt.xticks(x_precision, my_xticks)
plt.plot(x_precision, y_precision,'ro')

plt.figure("F1 Score")
x_F1Score = np.array([0,2,4,6,8,10])
y_F1Score = np.array([fbeta_score3u,fbeta_score5u,fbeta_score10u,fbeta_score3d,fbeta_score5d,fbeta_score10d])
my_xticks = ['k3_uni','k5_uni','k10_uni','k3_dis','k5_dis','k10_dis']
plt.xticks(x_F1Score, my_xticks)
plt.plot(x_F1Score, y_F1Score,'ro')
plt.show()

'''
Based on the accuracy and f1-score, the perfomance of model is the same for all the hyperparameters
,hence we choose k=3 and uniform.
'''
# In[33]:
# Evaluating Test Dataset for galaxy with the chosen hyperparameter (i.e. k=3 and weight=uniform)
Y_Test_pred_k3u = knn_3u.predict(X_test_galaxy)
Y_Test_pred_prob_k3u = knn_3u.predict_proba(X_test_galaxy)
acc_galaxy_test_k3u = accuracy_score(Y_test_galaxy, Y_Test_pred_k3u )
print("Correct classification rate for Test dataset with k:3 & weight=uniform = "+str(acc_galaxy_test_k3u*100)+"%")
cm_galaxy_test_k3u= confusion_matrix(Y_test_galaxy, Y_Test_pred_k3u )

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix for max depth =7
plt.figure()
plot_confusion_matrix(cm_galaxy_test_k3u, classes= ['Smooth', 'Spiral'],
                      title='Confusion matrix, without normalization with k:3')
plt.figure()
plot_confusion_matrix(cm_galaxy_test_k3u, classes= ['Smooth', 'Spiral'], normalize=True,
                      title='Confusion matrix, with normalization with k:3')
plt.show()
# In[33]:
# In[33]:
print("       Bayes Naif models with hold-out set ")

# Scale data for train, validation (hold out), and test
# first method of discretization using MDLP
from mdlp.discretization import MDLP
mdlp = MDLP()
Xtrain_galaxy_MDLP = mdlp.fit_transform(X_train_galaxy, Y_train_galaxy)
Xtest_galaxy_MDLP = mdlp.transform(X_test_galaxy, Y_test_galaxy)
Xvalid_galaxy_MDLP = mdlp.transform(X_valid_galaxy, Y_valid_galaxy)

# In[33]:
# Second method of discretization using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
Xtrain_galaxy_unsupervised = scaler.fit_transform(X_train_galaxy)
Xtest_galaxy_unsupervised = scaler.transform(X_test_galaxy)
Xvalid_galaxy_unsupervised= scaler.transform(X_valid_galaxy)
# In[33]:

# Bayes naïf gaussien with 2 different parameters i.e. 
# 1. priors = probaility of each class

# In[33]:
from sklearn.naive_bayes import GaussianNB
GaussianClassifier_priors = GaussianNB(priors=[0.48,0.52])
print("Bayes naïf gaussien & priors = probaility of each class")
GaussianClassifier_priors=GaussianClassifier_priors.fit(X_train_galaxy, Y_train_galaxy)
Y_validation_pred_Gaussian_priors = GaussianClassifier_priors.predict(X_valid_galaxy)
Y_validation_pred_prob_Gaussian_priors = GaussianClassifier_priors.predict_proba(X_valid_galaxy)

# In[33]:
# Bayes naïf gaussien with 2 different parameters i.e. 
# 2.without priors = probaility of each class
GaussianClassifier_without_priors = GaussianNB()
print("Bayes naïf gaussien without priors = probaility of each class")
GaussianClassifier_without_priors=GaussianClassifier_without_priors.fit(X_train_galaxy, Y_train_galaxy)
Y_validation_pred_Gaussian_without_priors = GaussianClassifier_without_priors.predict(X_valid_galaxy)
Y_validation_pred_prob_Gaussian_without_priors = GaussianClassifier_without_priors.predict_proba(X_valid_galaxy)
# In[33]:

# Bayes naïf multinomial with three different parameters i.e. 
# 1. priors = probaility of each class
# 2. MDLP discretization
# 3. unsupervised discretization
from sklearn.naive_bayes import MultinomialNB
print("Bayes naïf multinomial & priors = probaility of each class")
MultinomialNBClassifier_priors = MultinomialNB(class_prior=[0.48,0.52], fit_prior=True)
MultinomialNBClassifier_priors=MultinomialNBClassifier_priors.fit(Xtrain_galaxy_MDLP,Y_train_galaxy)
Y_validation_pred_Multinomial_priors = MultinomialNBClassifier_priors.predict(Xvalid_galaxy_MDLP)
Y_validation_pred_prob_Multinomial_priors = MultinomialNBClassifier_priors.predict_proba(Xvalid_galaxy_MDLP)

# In[33]:
MultinomialNBClassifier_discret=MultinomialNB()
print("Bayes naïf multinomial & MDLP discretization")
MultinomialNBClassifier_discret_MDLP=MultinomialNBClassifier_discret.fit(Xtrain_galaxy_MDLP, Y_train_galaxy)
Y_validation_pred_Multinomial_discret_MDLP = MultinomialNBClassifier_discret_MDLP.predict(Xvalid_galaxy_MDLP)
Y_validation_pred_prob_Multinomial_discret_MDLP = MultinomialNBClassifier_discret_MDLP.predict_proba(Xvalid_galaxy_MDLP)

print("Bayes naïf multinomial & unsupervised discretization")
MultinomialNBClassifier_discret_unsupervised=MultinomialNBClassifier_discret.fit(Xtrain_galaxy_unsupervised, Y_train_galaxy)
Y_validation_pred_Multinomial_discret_unsupervised = MultinomialNBClassifier_discret_unsupervised.predict(Xvalid_galaxy_unsupervised)
Y_validation_pred_prob_Multinomial_discret_unsupervised = MultinomialNBClassifier_discret_unsupervised.predict_proba(Xvalid_galaxy_unsupervised)

acc_galaxy_validation_Gaussian_priors = accuracy_score(Y_valid_galaxy, Y_validation_pred_Gaussian_priors )
print("Correct classification rate for validation dataset with Gaussian and priors = "+str(acc_galaxy_validation_Gaussian_priors*100)+"%")
acc_galaxy_validation_Gaussian_without_priors = accuracy_score(Y_valid_galaxy, Y_validation_pred_Gaussian_priors )
print("Correct classification rate for validation dataset with Gaussian without priors = "+str(acc_galaxy_validation_Gaussian_without_priors*100)+"%")
acc_galaxy_validation_multinomial_priors = accuracy_score(Y_valid_galaxy, Y_validation_pred_Multinomial_priors )
print("Correct classification rate for validation dataset with multinomial priors = "+str(acc_galaxy_validation_multinomial_priors*100)+"%")
acc_galaxy_validation_multinomial_MDLP = accuracy_score(Y_valid_galaxy, Y_validation_pred_Multinomial_discret_MDLP )
print("Correct classification rate for validation dataset with multinomial MDLP = "+str(acc_galaxy_validation_multinomial_MDLP*100)+"%")
acc_galaxy_validation_multinomial_unsupervised = accuracy_score(Y_valid_galaxy, Y_validation_pred_Multinomial_discret_unsupervised )
print("Correct classification rate for validation dataset with multinomial unsupervised = "+str(acc_galaxy_validation_multinomial_unsupervised*100)+"%")

print("Gaussian with priors ")
precision_gaussian_priors,_,fbeta_score_gaussian_priors,_=precision_recall_fscore_support(Y_valid_galaxy,Y_validation_pred_Gaussian_priors,average="weighted")
print ("precision is {} and F1-score is {}".format(precision_gaussian_priors,fbeta_score_gaussian_priors))
print("Gaussian  without priors")
precision_gaussian_without_priors,_,fbeta_score_gaussian_without_priors,_=precision_recall_fscore_support(Y_valid_galaxy,Y_validation_pred_Gaussian_without_priors,average="weighted")
print ("precision is {} and F1-score is {}".format(precision_gaussian_without_priors,fbeta_score_gaussian_without_priors))
print("\n Bayes naïf multinomial & priors")
precision_multinomial_priors,_,fbeta_score_multinomial_priors,_=precision_recall_fscore_support(Y_valid_galaxy,Y_validation_pred_Multinomial_priors,average="weighted")
print ("precision is {} and F1-score is {}".format(precision_multinomial_priors,fbeta_score_multinomial_priors))
print("\n Bayes naïf multinomial & MDLP")
precision_multinomial_MDLP,_,fbeta_score_multinomial_MDLP,_=precision_recall_fscore_support(Y_valid_galaxy,Y_validation_pred_Multinomial_discret_MDLP,average="weighted")
print ("precision is {} and F1-score is {}".format(precision_multinomial_MDLP,fbeta_score_multinomial_MDLP))
print("\n Bayes naïf multinomial & unsupervised")
precision_multinomial_unsupervised,_,fbeta_score_multinomial_unsupervised,_=precision_recall_fscore_support(Y_valid_galaxy,Y_validation_pred_Multinomial_discret_unsupervised,average="weighted")
print ("precision is {} and F1-score is {}".format(precision_multinomial_unsupervised,fbeta_score_multinomial_unsupervised))

plt.figure("Accuracy")
x_precision = np.array([0,2,4,6,8])
y_precision = np.array([precision_gaussian_priors,precision_gaussian_without_priors,precision_multinomial_priors,precision_multinomial_MDLP,precision_multinomial_unsupervised])
my_xticks = ['gaussian_priors','gaussian_without_priors','multinomial','multinomial_MDLP','multinomial_unsupervised']
plt.xticks(x_precision, my_xticks)
plt.plot(x_precision, y_precision,'ro')

plt.figure("F1 Score")
x_F1Score = np.array([0,2,4,6,8])
y_F1Score = np.array([fbeta_score_gaussian_priors,fbeta_score_gaussian_without_priors,fbeta_score_multinomial_priors,fbeta_score_multinomial_MDLP,fbeta_score_multinomial_unsupervised])
my_xticks = ['gaussian_priors','gaussian_without_priors','multinomial','multinomial_MDLP','multinomial_unsupervised']
plt.xticks(x_F1Score, my_xticks)
plt.plot(x_F1Score, y_F1Score,'ro')
plt.show()

# In[33]:
# Evaluating Test Dataset for galaxy with the chosen hyperparameter 
#(i.e.the best results with multinomial MDLP during validation step,
# gives : precision is 0.8292048531629597 and F1-score is 0.8288554653035476
Y_Test_pred_multinomial = MultinomialNBClassifier_discret_MDLP.predict(Xtest_galaxy_MDLP)
acc_galaxy_test_multinomial = accuracy_score(Y_test_galaxy, Y_Test_pred_multinomial )
print("Correct classification rate for Test dataset with gaussian = "+str(acc_galaxy_test_multinomial*100)+"%")
cm_galaxy_test_multinomial= confusion_matrix(Y_test_galaxy, Y_Test_pred_multinomial )

np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cm_galaxy_test_multinomial, classes= ['Smooth', 'Spiral'],
                      title='Confusion matrix, without normalization with multinomial')
plt.figure()
plot_confusion_matrix(cm_galaxy_test_multinomial, classes= ['Smooth', 'Spiral'], normalize=True,
                      title='Confusion matrix, with normalization with multinomial')
plt.show()