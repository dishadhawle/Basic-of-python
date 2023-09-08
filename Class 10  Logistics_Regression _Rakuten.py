#!/usr/bin/env python
# coding: utf-8

# Import Libraries 

# step 1:-import the libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[2]:


### step 2:-Load the file


# Load and review data

# In[3]:


car_df=pd.read_csv("auto-mpg.csv")


# In[4]:


car_df.shape


# In[5]:


car_df.sample(10)


# In[6]:


car_df.drop("car name",axis=1,inplace=True)


# In[7]:


# Also replacing the categorical var with actual values
car_df["origin"]=car_df["origin"].replace({1: 'america',2: 'europe',3: 'asia'})
car_df.sample(10)


# In[8]:


### one hot encoding
car_df=pd.get_dummies(car_df,columns=['origin'])  ### one hot encoding
car_df.sample(10)


# In[9]:


car_df.isnull().sum()


# In[10]:


car_df.dtypes


# ## Dealing with Missing Values

# In[11]:


# quick summary of the data columns
car_df.describe()


# In[12]:


car_df.describe(include="all")


# In[13]:


car_df.info()


# In[14]:


#hp is missing cause it does not seem to be recognized as a numerial column!
car_df.dtypes


# In[15]:


#isdigit()? on 'horsepower'
hpIsDigit=pd.DataFrame(car_df.horsepower.str.isdigit())  #if the string is made of digit store True else 
#print isDigit=False!
car_df[hpIsDigit['horsepower']==False]


# In[16]:


car_df["horsepower"]=car_df["horsepower"].replace("?",np.nan)
car_df["horsepower"]=car_df["horsepower"].astype(float)


# In[17]:


median1=car_df["horsepower"].median()
median1


# In[18]:


car_df["horsepower"].replace(np.nan,median1,inplace=True)


# In[19]:


car_df[hpIsDigit['horsepower']==False]


# In[20]:


car_df.dtypes


# In[21]:


### Duplicates?
duplicate=car_df.duplicated() 
duplicate.sum()


# ## Bivariate Plots

# In[22]:


sns.pairplot(car_df,diag_kind="kde")


# ## split Data

# In[23]:


x= car_df.drop(['mpg'], axis=1)
y= car_df[['mpg']]


# In[24]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)


# ## Fit Linear Model

# In[25]:


mode1=LinearRegression()
model_1.model_1fit(x_train, y_train)


# In[26]:


model_1.score(x_train,y_train)


# In[27]:


#out of sample score(R^2)

model_1.score(x_test,y_test)


# In[38]:


from sklearn.preprocessing import polynomialFeatures 
from sklearn import linear_model

poly=polynomialFeatures(degree=2,interaction_only=True)
x_train2=poly.fit_transform(x_train)
x_test2=poly.fit_transform(x_test)

poly_clf = linear_model.LinearRegression()

poly_clf.fit(x_train2, y_train)

#y_pred=poly_clf.predict(x_test2)

#print(y_pred)

#In sample (training)R^2 will always improve with the number of variables!
print(poly_clf.score(x_train2,y_train))



# In[ ]:


#out off sample (testing)R^2 is our measure of sucess and does improve
print (poly_clf.score(x_test2,y_test))

