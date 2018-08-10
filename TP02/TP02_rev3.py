
# coding: utf-8

# In[3]:


# Imports
import numpy as np


# In[4]:


# Première étape: créer un ensemble de validation pour le dataset des pourriels

# Load data from file (converted from nominal to numerical (binary approach))
num_features_spam = 57
num_features_galaxy = 75

data_spam = np.loadtxt("spam.csv", delimiter=",", skiprows=0)
data_galaxy = np.loadtxt("galaxy_feature_vectors.csv", delimiter=",", skiprows=0)


# In[5]:


data_spam


# In[6]:


data_galaxy


# In[7]:


# Define the training set
X_data_spam  = data_spam[:,0:num_features_spam]
Y_data_spam  = data_spam[:,num_features_spam] # last column = class labels

X_data_galaxy = data_galaxy[:,0:num_features_galaxy]
Y_data_galaxy = data_galaxy[:,num_features_galaxy] # last column = class labels


# In[60]:


# For Gaussian models, we're going to need the proportion of each label in each dataset
galaxy_smooth_ratio = (Y_data_galaxy == 0).sum() / len(Y_data_galaxy)
galaxy_spiral_ratio = (Y_data_galaxy == 1).sum() / len(Y_data_galaxy)

spam_spam_ratio = (Y_data_spam == 0).sum() / len(Y_data_spam)
spam_nonspam_ratio = (Y_data_spam == 1).sum() / len(Y_data_spam)

print("Proportion of smooth galaxies in galaxy dataset:", round(galaxy_smooth_ratio, 4))
print("Proportion of spiral galaxies in galaxy dataset:", round(galaxy_spiral_ratio, 4), "\n")
print("Proportion of emails marked as spam in spam dataset:", round(spam_spam_ratio, 4))
print("Proportion of emails marked as non-spam in spam dataset:", round(spam_nonspam_ratio, 4))


# In[8]:


X_data_spam


# In[9]:


X_data_galaxy


# In[10]:


Y_data_spam


# In[11]:


Y_data_galaxy


# In[12]:


# Using hold-out evaluation
from sklearn.model_selection import train_test_split

# - Split the data into train and valid, holding 20% of the data into valid
# - We are suffling the data to avoid ordered data by labels
# - Stratification means that the train_test_split method returns
#   training and test subsets that have the same proportions of class labels as the input dataset.
# - Random_state is desirable for reproducibility.

X_train_spam, X_valid_spam, Y_train_spam, Y_valid_spam = train_test_split(
    X_data_spam, Y_data_spam, test_size=0.2, random_state=0, shuffle=True, stratify=Y_data_spam
)

X_train_galaxy, X_valid_galaxy, Y_train_galaxy, Y_valid_galaxy = train_test_split(
    X_data_galaxy, Y_data_galaxy, test_size=0.2, random_state=0, shuffle=True, stratify=Y_data_galaxy
)


# In[13]:


print("X_train_spam length:", len(X_train_spam))
print("X_valid_spam length:", len(X_valid_spam))
print("Y_train_spam length:", len(Y_train_spam))
print("Y_valid_spam length:", len(Y_valid_spam))
print("")
print("X_train_galaxy length:", len(X_train_galaxy))
print("X_valid_galaxy length:", len(X_valid_galaxy))
print("Y_train_galaxy length:", len(Y_train_galaxy))
print("Y_valid_galaxy length:", len(Y_valid_galaxy))


# In[14]:


# Let's verify we have the statify option work
# We should have the same proportion of non-spam and spam labels inside both data sets
print(Y_train_spam.mean())
print(Y_valid_spam.mean())

print(Y_train_galaxy.mean())
print(Y_valid_galaxy.mean())


# In[15]:


# StratifiedKFold
#from sklearn.model_selection import StratifiedKFold

#skf = StratifiedKFold(n_splits=16)

#for train_index, test_index in skf.split(X_train_spam, Y_train_spam):
#    print(len(Y_train_spam[train_index]), Y_train_spam[train_index].mean())

############# Inutile?


# In[16]:


# Produisez un code source permettant d’entraîner un modèle afin de classifier les pourriels avec l’aide des
# trois algorithmes vus en classe, soit :
# 1. un arbre de décision
# 2. Bayes naïf (nous allons devoir utiliser MultinomialNB)
# 3. KNN

import graphviz
from sklearn import tree

# 1. Arbre de décision
# Train the Decision Tree with the training set
model = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=10)
model = model.fit(X_train_galaxy, Y_train_galaxy)


# In[17]:


# Visualize the tree in jupyter and save it in a PNG file
dot_data = tree.export_graphviz(model, out_file=None, 
                         feature_names = [
                             'Couleur moyenne du centre [0]',
                             'Couleur moyenne du centre [1]',
                             'Couleur moyenne du centre [2] ',
                             'Couleur moyenne [0]',
                             'Couleur moyenne [1]',
                             'Couleur moyenne [2]',
                             'Standard Deviation [0]',
                             'Standard Deviation [1]',
                             'Standard Deviation [2]',
                             'Distribution Kurtosis [0]',
                             'Distribution Kurtosis [1]',
                             'Distribution Kurtosis [2]',
                             'Distribution normale asymétrique [0]',
                             'Distribution normale asymétrique [1]',
                             'Distribution normale asymétrique [2]',
                             'Coefficient Gini [0]',
                             'Coefficient Gini [1]',
                             'Coefficient Gini [2]',
                             'Excentricité',
                             'Largeur',
                             'Hauteur',
                             'Somme',
                             'Entropie',
                             'Chiralité',
                             'Aire de l’ellipse',
                             'Aire box-to-image',
                             'Décalage du centre (offset)',
                             'Rayon de la lumière [0] ',
                             'Rayon de la lumière [1] ',
                             'Nombre de labels',
                             'Distribution Kurtosis [0] ',
                             'Distribution Kurtosis [1] ',
                             'Distribution Kurtosis [2] ',
                             'Distribution normale asymétrique [0] ',
                             'Distribution normale asymétrique [1] ',
                             'Distribution normale asymétrique [2] ',
                             'Coefficient Gini [0]',
                             'Coefficient Gini [1]',
                             'Coefficient Gini [2]',
                             'Distribution Kurtosis image noir et blanc',
                             'Distribution normale asymétrique image noir et blanc',
                             'Coefficient Gini image noir et blanc',
                             'Couleur du centre [0]',
                             'Couleur du centre [1]',
                             'Couleur du centre [2]',
                             'Couleur moyenne [0]',
                             'Couleur moyenne [1]',
                             'Couleur moyenne [2]',
                             'Couleur moyenne du centre [0]',
                             'Couleur moyenne du centre [1',
                             'Couleur moyenne du centre [2',
                             'Couleur du centre ',
                             'Rapport couleur du centre / moyenne de gris',
                             'Moments de l’image',
                             'Moments de l’image',
                             'Moments de l’image',
                             'Moments de l’image',
                             'Moments de l’image',
                             'Moments de l’image',
                             'Moments de l’image',
                             'Moments de l’image',
                             'Moments de l’image',
                             'Moments de l’image',
                             'Moments de l’image',
                             'Moments de l’image',
                             'Moments de l’image',
                             'Moments de l’image',
                             'Moments de l’image',
                             'Moments de l’image',
                             'Moments de l’image',
                             'Moments de l’image',
                             'Moments de l’image',
                             'Moments de l’image',
                             'Moments de l’image',
                             'Moments de l’image',
                         ],  
                         class_names = ['Smooth', 'Spiral'],
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render("Simpsons_Data") 
graph


# In[18]:


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


# In[19]:


# Première étape: créer un ensemble de validation pour le dataset des pourriels

# Load data from file (converted from nominal to numerical (binary approach))

'''
Galaxy dataset lables: smooth=0, spiral=1
Emails dataset labels: spam (1) » ou « non-spam (0)

'''
#num_features_spam = 57
#num_features_galaxy = 75
#data_spam = np.loadtxt("spam.csv", delimiter=",")
#data_galaxy = np.loadtxt("galaxy_feature_vectors.csv", delimiter=",")

########## C'est pas déjà fait?


# In[20]:


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


# In[21]:


# Let's verify we have the statify option work
# We should have the same proportion of non-spam and spam labels inside both data sets
print(Y_train_spam.mean())
print(Y_valid_spam.mean())

print(Y_train_galaxy.mean())
print(Y_valid_galaxy.mean())
count_ones=0
count_zeros=0
for i in Y_train_galaxy:
    if i==0:
        count_zeros+=1
    else:
        count_ones+=1
print ("number of smooth galaxies in train dataset is {} and spiral galaxies {}".format(count_zeros,count_ones))
count_ones=0
count_zeros=0
for i in Y_test_galaxy:
    if i==0:
        count_zeros+=1
    else:
        count_ones+=1
print ("number of smooth galaxies in test dataset is {} and spiral galaxies {}".format(count_zeros,count_ones))

count_ones=0
count_zeros=0
for i in Y_valid_galaxy:
    if i==0:
        count_zeros+=1
    else:
        count_ones+=1
print ("number of smooth galaxies in validation dataset is {} and spiral galaxies {}".format(count_zeros,count_ones))

count_ones=0
count_zeros=0
for i in Y_train_spam:
    if i==0:
        count_zeros+=1
    else:
        count_ones+=1
print ("number of non-spam emails in train dataset is {} and spam emails {}".format(count_zeros,count_ones))
count_ones=0
count_zeros=0
for i in Y_test_spam:
    if i==0:
        count_zeros+=1
    else:
        count_ones+=1
print ("number of non-spam emails in test dataset is {} and spam emails {}".format(count_zeros,count_ones))
count_ones=0
count_zeros=0
for i in Y_valid_spam:
    if i==0:
        count_zeros+=1
    else:
        count_ones+=1
print ("number of non_spam emails in valid dataset is {} and spam emails {}".format(count_zeros,count_ones))


# In[22]:


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


# In[ ]:


#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#
#
#                                   1: Galaxy
#
#
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################


# In[ ]:


#################################################################################################
#
#                             1.1: Holdout method
#
#################################################################################################


# In[23]:


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


# In[24]:


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


# In[25]:


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


# In[26]:


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


# In[27]:


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


# In[28]:


acc_galaxy_validation_max_depth10 = accuracy_score(Y_valid_galaxy, Y_validation_pred_max_depth10 )
print("Correct classification rate for validation dataset with max depth 10 = "+str(acc_galaxy_validation_max_depth10*100)+"%")

acc_galaxy_validation_max_depth5 = accuracy_score(Y_valid_galaxy, Y_validation_pred_max_depth5 )
print("Correct classification rate for validation dataset with max depth 5 = "+str(acc_galaxy_validation_max_depth5*100)+"%")

acc_galaxy_validation_max_depth3 = accuracy_score(Y_valid_galaxy, Y_validation_pred_max_depth3 )
print("Correct classification rate for validation dataset with max depth 3 = "+str(acc_galaxy_validation_max_depth3*100)+"%")

acc_galaxy_validation_max_depth_None = accuracy_score(Y_valid_galaxy, Y_validation_pred_max_depth_None )
print("Correct classification rate for validation dataset with max depth None = "+str(acc_galaxy_validation_max_depth_None*100)+"%")


# In[29]:


from sklearn.metrics import precision_recall_fscore_support


# In[30]:


# evaluating different hyperparameters for Decision tree models(i.e. Accuracy & F1_Score)
print("max_depth = 10")
# precision_recall_fscore allows us to get both the precision and f1 score in the same call
precision10,_,fbeta_score10,_=precision_recall_fscore_support(Y_valid_galaxy,Y_validation_pred_max_depth10,average="weighted")
print ("precision is {} and F1-score is {}".format(precision10,fbeta_score10))

print("\nmax_depth = 5")
precision5,_,fbeta_score5,_=precision_recall_fscore_support(Y_valid_galaxy,Y_validation_pred_max_depth5,average="weighted")
print ("precision is {} and F1-score is {}".format(precision5,fbeta_score5))

print("\nmax_depth = 3")
precision3,_,fbeta_score3,_=precision_recall_fscore_support(Y_valid_galaxy,Y_validation_pred_max_depth3,average="weighted")
print ("precision is {} and F1-score is {}".format(precision3,fbeta_score3))

print("\nmax_depth = None")
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


# In[31]:


# Evaluating Test Dataset for galaxy with the chosen hyperparameter (i.e. Without max_depth)
Y_Test_pred_max_depth_None = model_depth_None.predict(X_test_galaxy)

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


# In[32]:


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


#Classify using k-NN for k=3,5,10 and weight=uniform
from sklearn.neighbors import KNeighborsClassifier
print("k=3 and weight= uniform")
knn_3u = KNeighborsClassifier(n_neighbors=3, weights="uniform")
knn_3u.fit(Xtrain_galaxy_s, Y_train_galaxy)

Y_validation_pred_3u = knn_3u.predict(X_valid_galaxy)

print("k=5 and weight= uniform")
knn_5u = KNeighborsClassifier(n_neighbors=5, weights="uniform")
knn_5u.fit(Xtrain_galaxy_s, Y_train_galaxy)

Y_validation_pred_5u = knn_5u.predict(X_valid_galaxy)

print("k=10 and weight= uniform")
knn_10u = KNeighborsClassifier(n_neighbors=10, weights="uniform")
knn_10u.fit(Xtrain_galaxy_s, Y_train_galaxy)

Y_validation_pred_10u = knn_10u.predict(X_valid_galaxy)

acc_galaxy_validation_3u = accuracy_score(Y_valid_galaxy, Y_validation_pred_3u )
print("Correct classification rate for validation dataset with k=3 and weight: uniform = "+str(acc_galaxy_validation_3u*100)+"%")
acc_galaxy_validation_5u = accuracy_score(Y_valid_galaxy, Y_validation_pred_5u )
print("Correct classification rate for validation dataset with k=5 and weight: uniform = "+str(acc_galaxy_validation_5u*100)+"%")
acc_galaxy_validation_10u = accuracy_score(Y_valid_galaxy, Y_validation_pred_10u )
print("Correct classification rate for validation dataset with k=10 and weight: uniform = "+str(acc_galaxy_validation_10u*100)+"%")


# In[34]:


#Classify using k-NN for k=3,5,10 and weight=distance
print("k=3 and weight=distance")
knn_3d = KNeighborsClassifier(n_neighbors=3, weights="distance")
knn_3d.fit(Xtrain_galaxy_s, Y_train_galaxy)

Y_validation_pred_3d = knn_3u.predict(X_valid_galaxy)

print("k=5 and weight=distance")
knn_5d = KNeighborsClassifier(n_neighbors=5, weights="distance")
knn_5d.fit(Xtrain_galaxy_s, Y_train_galaxy)

Y_validation_pred_5d = knn_5d.predict(X_valid_galaxy)

print("k=10 and weight=distance")
knn_10d = KNeighborsClassifier(n_neighbors=10, weights="distance")
knn_10d.fit(Xtrain_galaxy_s, Y_train_galaxy)

Y_validation_pred_10d = knn_10d.predict(X_valid_galaxy)

acc_galaxy_validation_3d = accuracy_score(Y_valid_galaxy, Y_validation_pred_3d )
print("Correct classification rate for validation dataset with k=3 and weight: distance = "+str(acc_galaxy_validation_3d*100)+"%")

acc_galaxy_validation_5d = accuracy_score(Y_valid_galaxy, Y_validation_pred_5d )
print("Correct classification rate for validation dataset with k=5 and weight: distance = "+str(acc_galaxy_validation_5d*100)+"%")

acc_galaxy_validation_10d = accuracy_score(Y_valid_galaxy, Y_validation_pred_10d )
print("Correct classification rate for validation dataset with k=10 and weight: distance = "+str(acc_galaxy_validation_10d*100)+"%")


# In[35]:


# evaluating different hyperparameters for Decision tree models(i.e. Accuracy & F1_Score)
print("k=10 and weight=uniform")
precision10u,_,fbeta_score10u,_=precision_recall_fscore_support(Y_valid_galaxy,Y_validation_pred_10u,average="weighted")
print ("precision is {} and F1-score is {}".format(precision10u,fbeta_score10u))
print("\nk=5 and weight=uniform")
precision5u,_,fbeta_score5u,_=precision_recall_fscore_support(Y_valid_galaxy,Y_validation_pred_5u,average="weighted")
print ("precision is {} and F1-score is {}".format(precision5u,fbeta_score5u))

print("\nk=3 and weight=uniform")
precision3u,_,fbeta_score3u,_=precision_recall_fscore_support(Y_valid_galaxy,Y_validation_pred_3u,average="weighted")
print ("precision is {} and F1-score is {}".format(precision3u,fbeta_score3u))

print("\nk=10 and weight=distance")
precision10d,_,fbeta_score10d,_=precision_recall_fscore_support(Y_valid_galaxy,Y_validation_pred_10d,average="weighted")
print ("precision is {} and F1-score is {}".format(precision10d,fbeta_score10d))

print("\nk=5 and weight=distance")
precision5d,_,fbeta_score5d,_=precision_recall_fscore_support(Y_valid_galaxy,Y_validation_pred_5d,average="weighted")
print ("precision is {} and F1-score is {}".format(precision5d,fbeta_score5d))

print("\nk=3 and weight=distance")
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


# In[36]:


# Evaluating Test Dataset for galaxy with the chosen hyperparameter (i.e. k=3 and weight=uniform)
Y_Test_pred_k3u = knn_3u.predict(X_test_galaxy)

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


# In[37]:


print("       Bayes Naif models with hold-out set ")

# Scale data for train, validation (hold out), and test
# first method of discretization using MDLP
from mdlp.discretization import MDLP
mdlp = MDLP()
Xtrain_galaxy_MDLP = mdlp.fit_transform(X_train_galaxy, Y_train_galaxy)
Xtest_galaxy_MDLP = mdlp.transform(X_test_galaxy, Y_test_galaxy)
Xvalid_galaxy_MDLP = mdlp.transform(X_valid_galaxy, Y_valid_galaxy)


# In[38]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
Xtrain_galaxy_unsupervised = scaler.fit_transform(X_train_galaxy)
Xtest_galaxy_unsupervised = scaler.transform(X_test_galaxy)
Xvalid_galaxy_unsupervised= scaler.transform(X_valid_galaxy)


# In[39]:


# Bayes naïf gaussien with three different parameters i.e. 
# 1. priors = probaility of each class

# In[33]:
from sklearn.naive_bayes import GaussianNB
GaussianClassifier_priors = GaussianNB(priors=[0.48,0.52])
print("Bayes naïf gaussien & priors = probaility of each class")
GaussianClassifier_priors=GaussianClassifier_priors.fit(X_train_galaxy, Y_train_galaxy)
Y_validation_pred_Gaussian_priors = GaussianClassifier_priors.predict(X_valid_galaxy)


# In[40]:


# Bayes naïf multinomial with three different parameters i.e. 
# 1. priors = probaility of each class
# 2. MDLP discretization
# 3. unsupervised discretization
from sklearn.naive_bayes import MultinomialNB
print("Bayes naïf multinomial & priors = probaility of each class")
MultinomialNBClassifier_priors = MultinomialNB(class_prior=[0.48,0.52], fit_prior=True)
MultinomialNBClassifier_priors=MultinomialNBClassifier_priors.fit(Xtrain_galaxy_MDLP,Y_train_galaxy)
Y_validation_pred_Multinomial_priors = MultinomialNBClassifier_priors.predict(Xvalid_galaxy_MDLP)


# In[41]:


MultinomialNBClassifier_discret=MultinomialNB()
print("Bayes naïf multinomial & MDLP discretization")
MultinomialNBClassifier_discret_MDLP=MultinomialNBClassifier_discret.fit(Xtrain_galaxy_MDLP, Y_train_galaxy)
Y_validation_pred_Multinomial_discret_MDLP = MultinomialNBClassifier_discret_MDLP.predict(Xvalid_galaxy_MDLP)

print("Bayes naïf multinomial & unsupervised discretization")
MultinomialNBClassifier_discret_unsupervised=MultinomialNBClassifier_discret.fit(Xtrain_galaxy_unsupervised, Y_train_galaxy)
Y_validation_pred_Multinomial_discret_unsupervised = MultinomialNBClassifier_discret_unsupervised.predict(Xvalid_galaxy_unsupervised)

acc_galaxy_validation_Gaussian = accuracy_score(Y_valid_galaxy, Y_validation_pred_Gaussian_priors )
print("Correct classification rate for validation dataset with Gaussian = "+str(acc_galaxy_validation_Gaussian*100)+"%")
acc_galaxy_validation_multinomial_priors = accuracy_score(Y_valid_galaxy, Y_validation_pred_Multinomial_priors )
print("Correct classification rate for validation dataset with multinomial priors = "+str(acc_galaxy_validation_multinomial_priors*100)+"%")
acc_galaxy_validation_multinomial_MDLP = accuracy_score(Y_valid_galaxy, Y_validation_pred_Multinomial_discret_MDLP )
print("Correct classification rate for validation dataset with multinomial MDLP = "+str(acc_galaxy_validation_multinomial_MDLP*100)+"%")
acc_galaxy_validation_multinomial_unsupervised = accuracy_score(Y_valid_galaxy, Y_validation_pred_Multinomial_discret_unsupervised )
print("Correct classification rate for validation dataset with multinomial unsupervised = "+str(acc_galaxy_validation_multinomial_unsupervised*100)+"%")

print("Gaussian ")
precision_gaussian,_,fbeta_score_gaussian,_=precision_recall_fscore_support(Y_valid_galaxy,Y_validation_pred_Gaussian_priors,average="weighted")
print ("precision is {} and F1-score is {}".format(precision_gaussian,fbeta_score_gaussian))
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
x_precision = np.array([0,2,4,6])
y_precision = np.array([precision_gaussian,precision_multinomial_priors,precision_multinomial_MDLP,precision_multinomial_unsupervised])
my_xticks = ['gaussian','multinomial','multinomial_MDLP','multinomial_unsupervised']
plt.xticks(x_precision, my_xticks)
plt.plot(x_precision, y_precision,'ro')

plt.figure("F1 Score")
x_F1Score = np.array([0,2,4,6])
y_F1Score = np.array([fbeta_score_gaussian,fbeta_score_multinomial_priors,fbeta_score_multinomial_MDLP,fbeta_score_multinomial_unsupervised])
my_xticks = ['gaussian','multinomial','multinomial_MDLP','multinomial_unsupervised']
plt.xticks(x_F1Score, my_xticks)
plt.plot(x_F1Score, y_F1Score,'ro')
plt.show()


# In[42]:


# Evaluating Test Dataset for galaxy with the chosen hyperparameter (i.e. Despite the better results with multinomial MDLP during validation step,
# but, with test dataset, model gaussian gives the best result!!! 48% vs 71%)
Y_Test_pred_gaussian = GaussianClassifier_priors.predict(X_test_galaxy)
acc_galaxy_test_gaussian = accuracy_score(Y_test_galaxy, Y_Test_pred_gaussian )
print("Correct classification rate for Test dataset with gaussian = "+str(acc_galaxy_test_gaussian*100)+"%")
cm_galaxy_test_gaussian= confusion_matrix(Y_test_galaxy, Y_Test_pred_gaussian )

np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cm_galaxy_test_gaussian, classes= ['Smooth', 'Spiral'],
                      title='Confusion matrix, without normalization with gaussian')
plt.figure()
plot_confusion_matrix(cm_galaxy_test_gaussian, classes= ['Smooth', 'Spiral'], normalize=True,
                      title='Confusion matrix, with normalization with gaussian')
plt.show()


# In[43]:


#################################################################################################
#
#                             1.2: KFold method
#
#################################################################################################


# In[44]:


# - Split the data into train and valid, holding 20% of the data into valid
# - We are suffling the data to avoid ordered data by labels
# - Stratification means that the train_test_split method returns
#   training and test subsets that have the same proportions of class labels as the input dataset.
# - Random_state is desirable for reproducibility.
from sklearn.model_selection import train_test_split

X_train_galaxy, X_test_galaxy, Y_train_galaxy, Y_test_galaxy = train_test_split(
    X_data_galaxy, Y_data_galaxy, test_size=0.2, random_state=0, shuffle=True, stratify=Y_data_galaxy
)


# In[ ]:


#################################################################################################
#
#                             1.2.1: KFold - Decision Tree
#
#################################################################################################


# In[146]:


# StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn import tree

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

max_depths = np.array([3, 5, 10, None])
for i in range(0, 4):
    print("Decision Tree with 10 folds (max_depth=" + str(max_depths[i]) + ")\n")
    sum_precision = 0;
    sum_f1_score = 0;
    fold_counter = 0
    for train_index, test_index in skf.split(X_train_galaxy, Y_train_galaxy):
        X_train, X_valid = X_train_galaxy[train_index], X_train_galaxy[test_index];
        Y_train, Y_valid = Y_train_galaxy[train_index], Y_train_galaxy[test_index];

        model = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=10, max_depth=max_depths[i])
        model = model.fit(X_train, Y_train)

        Y_valid_pred = model.predict(X_valid)
        
        precision,_,f1_score,_ = precision_recall_fscore_support(Y_valid, Y_valid_pred, average="weighted")
        
        fold_counter += 1
        sum_precision += precision;
        sum_f1_score += f1_score;
        
        print("Correct classification rate for fold #" + str(fold_counter) + ": " + str(round(precision * 100, 4)) + "% (max_depth=" + str(max_depths[i]) + ", f1_score=" + str(round(f1_score, 4)) + ")")
    print("\nCorrect classification rate: " + str(round((sum_precision / 10) * 100, 4)) + "%")
    print("F1 Score: " + str(round((sum_precision / 10), 4)) + "\n\n")



# In[ ]:


#################################################################################################
#
#                             1.2.2: KFold - Naive Bayes
#
#################################################################################################


# In[147]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn import tree
from mdlp.discretization import MDLP
from sklearn.naive_bayes import GaussianNB

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

sum_precision = 0
sum_f1_score = 0
fold_counter = 0

print("Naive Bayes Gaussian (priors) with 10 folds\n")

# Loop through the 10 folds
for train_index, test_index in skf.split(X_train_galaxy, Y_train_galaxy):
    X_train, X_valid = X_train_galaxy[train_index], X_train_galaxy[test_index];
    Y_train, Y_valid = Y_train_galaxy[train_index], Y_train_galaxy[test_index];

    model = GaussianNB(priors=[galaxy_smooth_ratio, galaxy_spiral_ratio])
    model = model.fit(X_train, Y_train)
    
    Y_valid_pred = model.predict(X_valid)
        
    precision,_,f1_score,_ = precision_recall_fscore_support(Y_valid, Y_valid_pred, average="weighted")
    
    fold_counter += 1
    sum_precision += precision;
    sum_f1_score += f1_score;
    print("Correct classification rate for fold #" + str(fold_counter) + ": " + str(round(precision * 100, 4)) + "%")



print("\nCorrect classification rate: " + str(round((sum_precision / 10) * 100, 4)) + "%")
print("F1 Score: " + str(round((sum_precision / 10), 4)))


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from mdlp.discretization import MDLP

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
mdlp = MDLP()

sum_precision = 0
sum_f1_score = 0
fold_counter = 0

print("Naive Bayes Multinomial (priors) with 10 folds\n")

# Loop through the 10 folds
for train_index, test_index in skf.split(X_train_galaxy, Y_train_galaxy):
    X_train, X_valid = X_train_galaxy[train_index], X_train_galaxy[test_index];
    Y_train, Y_valid = Y_train_galaxy[train_index], Y_train_galaxy[test_index];
    
    X_train_mdlp = mdlp.fit_transform(X_train, Y_train)
    X_valid_mdlp = mdlp.transform(X_valid, Y_valid)

    model = MultinomialNB(class_prior=[galaxy_smooth_ratio, galaxy_spiral_ratio], fit_prior=True)
    model = model.fit(X_train_mdlp, Y_train)
    
    Y_valid_pred = model.predict(X_valid_mdlp)
        
    precision,_,f1_score,_ = precision_recall_fscore_support(Y_valid, Y_valid_pred, average="weighted")
    
    sum_precision += precision;
    sum_f1_score += f1_score;
    fold_counter += 1
    print("Correct classification rate for fold #" + str(fold_counter) + ": " + str(round(precision * 100, 4)) + "%")

print("Correct classification rate: " + str(round((sum_precision / 10) * 100, 4)) + "%")
print("F1 Score: " + str(round((sum_precision / 10), 4)))


# In[ ]:


#################################################################################################
#
#                             1.2.3: KFold - KNN
#
#################################################################################################


# In[142]:


from sklearn import neighbors

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
mdlp = MDLP()

# we create an instance of Neighbours Classifier and train with the training dataset.
weights = 'uniform'
metric = 'euclidean'
algorithm = 'brute'

hyperparameter_k = [3, 5, 10]
    
# Loop through the 10 folds
for i in range(0, 3):
    print("KNN with 10 folds (k=" + str(hyperparameter_k[i]) + ")\n")
    
    sum_precision = 0
    sum_f1_score = 0
    fold_counter = 0
    
    for train_index, test_index in skf.split(X_train_galaxy, Y_train_galaxy):
        X_train, X_valid = X_train_galaxy[train_index], X_train_galaxy[test_index];
        Y_train, Y_valid = Y_train_galaxy[train_index], Y_train_galaxy[test_index];
        
        model = neighbors.KNeighborsClassifier(hyperparameter_k[i], weights=weights, algorithm=algorithm, metric=metric)
        model = model.fit(X_train, Y_train)

        Y_valid_pred = model.predict(X_valid)

        precision,_,f1_score,_ = precision_recall_fscore_support(Y_valid, Y_valid_pred, average="weighted")

        sum_precision += precision;
        sum_f1_score += f1_score;
        fold_counter += 1
        print("Correct classification rate for fold #" + str(fold_counter) + ": " + str(round(precision * 100, 4)) + "%")
        


    print("\nCorrect classification rate: " + str(round((sum_precision / 10) * 100, 4)) + "%")
    print("F1 Score: " + str(round((sum_precision / 10), 4)) + "\n\n")


# In[ ]:


#################################################################################################
#
#                             1.3: Leave-one-out method 
#
#################################################################################################




# In[ ]:


#################################################################################################
#
#                             1.3.1: Leave-one-out - Decision Tree
#
#################################################################################################
from sklearn.model_selection import  LeaveOneOut   
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score ,accuracy_score

max_depths = np.array([3, 5, 10, None])
scoring = ['accuracy', 'f1_weighted']
print("Decision Tree with LOO (max_depth= None \n")
model_None = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=10, max_depth=None)
scores_depth_None=cross_validate(model_None, X_train_galaxy, Y_train_galaxy, scoring=scoring,cv=LeaveOneOut(), return_train_score=False)

print("Decision Tree with LOO (max_depth=" + str(3) + ")\n")
model_3 = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=10, max_depth=3)
scores_depth_3=cross_validate(model_None, X_train_galaxy, Y_train_galaxy, scoring=scoring,cv=LeaveOneOut(), return_train_score=False)

print("Decision Tree with LOO (max_depth=" + str(5) + ")\n")
model_5 = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=10, max_depth=5)
scores_depth_5=cross_validate(model_None, X_train_galaxy, Y_train_galaxy, scoring=scoring,cv=LeaveOneOut(), return_train_score=False)

print("Decision Tree with LOO (max_depth=" + str(10) + ")\n")
model_10 = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=10, max_depth=10)
scores_depth_10=cross_validate(model_None, X_train_galaxy, Y_train_galaxy, scoring=scoring,cv=LeaveOneOut(), return_train_score=False)

plt.figure("Accuracy")
x_precision = np.array([0,3,7,10])
y_precision = np.array([scores_depth_None['test_accuracy'].mean(),scores_depth_3['test_accuracy'].mean(),scores_depth_5['test_accuracy'].mean(),scores_depth_10['test_accuracy'].mean()])
my_xticks = ['without max_depth','max_depth:3','max_depth:5','max_depth:10']
plt.xticks(x_precision, my_xticks)
plt.plot(x_precision, y_precision,'ro')

plt.figure("F1 Score")
x_F1Score = np.array([0,3,7,10])
y_F1Score = np.array([scores_depth_None['test_f1_weighted'].mean(),scores_depth_3['test_f1_weighted'].mean(),scores_depth_5['test_f1_weighted'].mean(),scores_depth_10['test_f1_weighted'].mean()])
my_xticks = ['without max_depth','max_depth:3','max_depth:5','max_depth:10']
plt.xticks(x_F1Score, my_xticks)
plt.plot(x_F1Score, y_F1Score,'ro')
plt.show()

print("test_accuracy_None= "+ scores_depth_None['test_accuracy'].mean())
print("test_accuracy_3= "+ scores_depth_3['test_accuracy'].mean())
print("test_accuracy_5= "+ scores_depth_5['test_accuracy'].mean())
print("test_accuracy_10= "+ scores_depth_10['test_accuracy'].mean())
print("f1_weighted_None= "+ scores_depth_None['test_f1_weighted'].mean())
print("f1_weighted_3= "+ scores_depth_3['test_f1_weighted'].mean())
print("f1_weighted_5= "+ scores_depth_5['test_f1_weighted'].mean())
print("f1_weighted_10= "+ scores_depth_10['test_f1_weighted'].mean())

# In[ ]:
# testing the best model
       
''' Evaluating Test Dataset for galaxy with the chosen hyperparameter (i.e. To be completed)
it depends on the results of validation for different max_depths. 
'''
# =============================================================================
# Y_Test_pred_max_depth_None = model_None.predict(X_test_galaxy)
# 
# acc_galaxy_test_max_depth_None = accuracy_score(Y_test_galaxy, Y_Test_pred_max_depth_None )
# print("Correct classification rate for Test dataset with max depth None = "+str(acc_galaxy_test_max_depth_None*100)+"%")
# cm_galaxy_test_max_depth_None = confusion_matrix(Y_test_galaxy, Y_Test_pred_max_depth_None )
# 
# np.set_printoptions(precision=2)
# 
# # Plot non-normalized confusion matrix for max depth =7
# plt.figure()
# plot_confusion_matrix(cm_galaxy_test_max_depth_None, classes= ['Smooth', 'Spiral'],
#                       title='Confusion matrix, without normalization without max_depth')
# plt.figure()
# plot_confusion_matrix(cm_galaxy_test_max_depth_None, classes= ['Smooth', 'Spiral'], normalize=True,
#                       title='Confusion matrix, with normalization without max_depth')
# plt.show()
# =============================================================================

# In[ ]:


#################################################################################################
#
#                             1.3.2: Leave-one-out - Naive Bayes
#
#################################################################################################


# In[ ]:


#################################################################################################
#
#                             1.3.3: Leave-one-out - KNN
#
#################################################################################################


# In[61]:


#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#
#
#                                   2: SPAM
#
#
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################


# In[ ]:


#################################################################################################
#
#                             2.1: Holdout method
#
#################################################################################################


# In[ ]:


#################################################################################################
#
#                             2.1.1: Holdout - Decision Tree
#
#################################################################################################


# In[ ]:


#################################################################################################
#
#                             2.1.2: Holdout - Naive Bayes
#
#################################################################################################


# In[ ]:


#################################################################################################
#
#                             2.1.3: Holdout - KNN
#
#################################################################################################


# In[ ]:


#################################################################################################
#
#                             2.2: KFold method
#
#################################################################################################


# In[ ]:


#################################################################################################
#
#                             2.2.1: KFold - Decision Tree
#
#################################################################################################


# In[ ]:


#################################################################################################
#
#                             2.2.2: KFold - Naive Bayes
#
#################################################################################################


# In[ ]:


#################################################################################################
#
#                             2.2.3: KFold - KNN
#
#################################################################################################


# In[ ]:


#################################################################################################
#
#                             2.3: Leave-one-out method
#
#################################################################################################


# In[ ]:


#################################################################################################
#
#                             2.3.1: Leave-one-out - Decision Tree
#
#################################################################################################


# In[ ]:


#################################################################################################
#
#                             2.3.2: Leave-one-out - Naive Bayes
#
#################################################################################################


# In[ ]:


#################################################################################################
#
#                             2.3.3: Leave-one-out - KNN
#
#################################################################################################


# In[ ]:


# TODO: Copy-paste galaxy stuff in there and change values for spam values


