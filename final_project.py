#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import requests
import json


# In[4]:


df = pd.read_csv("/Users/maryam/Desktop/Final Project/final_test.csv")


# In[5]:


df


# In[8]:


df.head()


# In[6]:


df.describe()


# In[7]:


df.info()


# In[9]:


df.isnull()


# In[11]:


df.isnull().sum()


# In[13]:


df.nunique()


# In[15]:


df['weight'].unique()


# In[16]:


df['height'].unique()


# In[18]:


df['age'].unique()


# In[24]:


df['size'].unique()


# In[36]:


sizee = df['size'].value_counts()
sizee


# In[34]:


df['size'].value_counts().plot.bar()


# In[40]:


x = list(sizee.index)
y = list(sizee)

fig, ax = plt.subplots()    
width = 0.75 # the width of the bars 
ind = np.arange(len(y))  # the x locations for the groups
ax.barh(ind, y, width, color="red")
ax.set_yticks(ind+width/2)
ax.set_yticklabels(x, minor=False)
for i, v in enumerate(y):
    ax.text(v + .25, i + .25, str(v), color='blue', fontweight='bold') #add value labels into bar
plt.title('title')
plt.xlabel('Count')
plt.ylabel('Size')
plt.show()


# In[47]:


df.isnull().sum()


# In[48]:


#remove all missing values from the dataset
df.dropna(inplace = True)
df.isnull().sum()


# In[49]:


X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values


# In[50]:


X


# In[51]:


y


# In[52]:


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# In[54]:


print("Train data shape of X = % s and Y = % s: "%(X_train.shape, y_train.shape))

print("Test data shape of X = % s and Y = % s: "%(
X_test.shape, y_test.shape))


# In[55]:


# Fitting Simple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[57]:


# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred


# In[61]:


#Calculating Mean Squared Error(mse)
mean_squared_error = np.mean((y_pred - y_test)**2)
print("Mean squared error on test set:", mean_squared_error)


# In[67]:


# printing values
print('Slope:' ,regressor.coef_)
print('Intercept:', regressor.intercept_)


# In[73]:


# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[15, 61,77]]))


# In[68]:


#another way to predict
from sklearn import linear_model
lm = linear_model.LinearRegression()
lm.fit(X, y)


# In[71]:


lm.predict([[15, 61, 77]])


# In[ ]:




