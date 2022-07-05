#!/usr/bin/env python
# coding: utf-8

# # Wine Quality Classification

# In[1]:


import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')


# In[2]:


wine = pd.read_csv(r"C:\Users\sreedev\Downloads\winequality-red.csv")


# In[3]:


wine


# In[4]:


wine.info()


# In[5]:


wine.isnull().sum()


# In[6]:


wine.columns


# In[7]:


# Defining x and y
y = wine[['quality']]
x = wine[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']]


# In[8]:


x.shape


# In[9]:


y.info()


# In[10]:


#Splitting data into train and test data.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 4321)


# In[11]:


x.head()


# In[12]:


#Building the model
# 1) Create a model object
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()


# In[13]:


# 2) Fitting the model object into training data
model = dt.fit(x_train, y_train)


# In[14]:


# Evlaute model accuracy 
# Perform prediction on test data
y_test['predicted quality'] = model.predict(x_test)


# In[15]:


y_test


# In[18]:


# Creating confusion matrix and Evaluating the model accuracy
from sklearn.metrics import confusion_matrix, accuracy_score

confusion_matrix(y_test['quality'], y_test['predicted quality'])


# All the diagonal values from top left to bottom right are the predicted accurate values.
# Other values are errors.

# In[22]:


accuracy = accuracy_score(y_test['quality'], y_test['predicted quality'])
accuracy


# In[ ]:




