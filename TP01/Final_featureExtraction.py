
# coding: utf-8

# ### Exercice 1

# In[44]:


# Imports General
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import matplotlib


# In[45]:


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler


# In[46]:


import csv
import math
import scipy.ndimage as nd
from scipy.stats.mstats import mquantiles, kurtosis, skew
from sklearn.preprocessing import LabelEncoder


# In[47]:


# Import OpenCV
import cv2


# In[48]:

train_path='C:/Users/RyanM/Desktop/Python Lab/Lab0/images_training/'
#valid_path='Simpsons-Train-Valid/Valid/'


# In[49]:


# Let's read each image filename in out list and store the corresponding image 
# Name to each class as X_train. Also, store the labels of galaxies as 1  and 0.
#
csvFile = open("GTI770_label_data_set.csv", "r")
count = 0

Y = np.ones(10000 , dtype=str)
X=['']*10000
choices = {'spiral' : 1, 'smooth': 0}
for i in csvFile:
    id_, name = i.split(',')                
    if id_ != "id":
        imagePath = train_path + id_ + '.jpg'
        if not os.path.exists(imagePath):
            continue
        X[count] = id_ + '.jpg'
        Y[count]=  choices.get(name.rstrip(), 99)
        count += 1
        if count == 10000:
            break


# In[50]:


#La matrice X a été divisée en 2 matrices avec un ratio de 70% (X_train)
#et 30% (X_test). Aussi, la matrice Y a été divisée en deux matrices de Y_train et Y_test.

from sklearn.model_selection import train_test_split
X_train, X_test1, Y_train, Y_test1 = train_test_split(X, Y, test_size=0.4,stratify=Y)
 
# In[51]:
X_test, X_validation, Y_test, Y_validation = train_test_split(X_test1, Y_test1, test_size=0.5,stratify=Y_test1) 

# In[51]:


''' Create empty feature vectors to store the features that we will extract from images
feature #1: Maximum values of Red and Blue channels for cropped image of 64 x 64
feature #2: Red to Blue color intensity ratio for cropped image of 64 x 64
feature #3: Black to White pixel ratio for cropped image of 64 x 64

In the feature_vector, the first two columns are maximum red and maximum blue color
and in the third column, 
'''
num_features = 6

# If we want to add the label at the end of the feature vector, we must add a dimension
num_features = num_features + 1

feature_vector = np.zeros( (len(X_train), num_features) )


# In[20]:


#finalColorMatrix=np.ones((200,2),dtype=int)
print(len(feature_vector))
img_count = 0
# Loop for each image in the list
for file_name in X_train:
    try:
        # Read the image
        img_color = cv2.imread(train_path+file_name,-1)
        # Create a gray scale of image
        img_grey=cv2.cvtColor(img_color,cv2.COLOR_BGR2GRAY)  
        ret,img_binary = cv2.threshold(img_grey,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # create a matrix to store color RGB pixels of each image
        colorMatrix=np.ones((4096,2),dtype=int)
        # Initialize variables with zero 
        n_white_pix=0
        n_black_pix=0
        index=0
        print ("Processing image " +str(img_count+1)+" "+file_name ) 
        
  
        for i in range(180,244):
              for j in range(180,244):     
                    red = img_color[i][j][2]
                    gre = img_color[i][j][1]
                    blu = img_color[i][j][0]
                    colorMatrix[index,0]=red
                    colorMatrix[index,1]=blu
                    index+=1              
                    if img_binary[i,j]==255:
                        n_white_pix+=1
                    else:
                        n_black_pix+=1
        '''
        # Store the features values in the columns of the feature (matrix) vector
        # feature 1 ,2 , 3 and 4
        '''
        redColumnAverage=np.mean(colorMatrix[:,0]) #mean of red colmun
        blueColumnAverage=np.mean(colorMatrix[:,1]) #mean of blue colmun   
        red2blueRatio=redColumnAverage/blueColumnAverage  # ratio of red to blue for each image          
        black2whitePixelRatio=n_black_pix/n_white_pix # ratio of black pixel number to white pixel number
          
        '''feature morpholgy #5 et #6
        contour feature detection'''
        # Blurring features
        img_blurred=cv2.pyrMeanShiftFiltering(img_color,61,181)
        # converting into gray scale
        img_grey=cv2.cvtColor(img_blurred,cv2.COLOR_BGR2GRAY)
        # convert to binary (threshold =245)
        ret,img_binary_grey = cv2.threshold(img_grey,245,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # find contours using openCV
        _,contours,_=cv2.findContours(img_binary_grey,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        # drawing contours
        cv2.drawContours(img_color,contours,-1,(130,255,255),3)
            
        ## find best contour for image
        best_contour_ma = 0
        best_contour_MA = 0
        
        for i, contour in enumerate(contours):
            try: 
                (x,y),(MA,ma),angle=cv2.fitEllipse(contour)
            
                print("Contour", i)
        
                if (MA * ma > best_contour_ma * best_contour_MA):
                    best_contour_ma = ma
                    best_contour_MA = MA
            except:
                print('Couldn\'t fit ellipse from contour');
                        
        #building feature vector
        feature_vector[img_count] = [np.max(colorMatrix[:,0]), np.max(colorMatrix[:,1]),red2blueRatio,black2whitePixelRatio,best_contour_ma/best_contour_MA, angle, Y_train[img_count]]
                 
        img_count += 1
    except:
        print("float division by zero")
        
       
    # Save the extracted feature vector into a text file.
np.savetxt("feature_vector_train.csv", feature_vector, delimiter=",")
        
print("number of countorurs %d" % len(contours))

print("Feature extraction ended successfully")

# In[54]:









