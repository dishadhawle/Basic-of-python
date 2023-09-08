#!/usr/bin/env python
# coding: utf-8

# Import Libraries 

# step 1:-import the libraries

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[9]:


### step 2:-Load the file


# Load and review data

# In[10]:


car_df=pd.read_csv("auto-mpg.csv")


# In[11]:


car_df.shape


# In[12]:


car_df.sample(10)


# In[13]:


car_df.drop("car name",axis=1,inplace=True)


# In[14]:


# Also replacing the categorical var with actual values
car_df["origin"]=car_df["origin"].replace({1: 'america',2: 'europe',3: 'asia'})
car_df.sample(10)


# In[15]:


### one hot encoding
car_df=pd.get_dummies(car_df,columns=['origin'])  ### one hot encoding
car_df.sample(10)


# In[16]:


car_df.isnull().sum()


# In[17]:


car_df.dtypes


# ## Dealing with Missing Values

# In[18]:


# quick summary of the data columns
car_df.describe()


# In[19]:


car_df.describe(include="all")


# In[20]:


car_df.info()


# In[21]:


#hp is missing cause it does not seem to be recognized as a numerial column!
car_df.dtypes


# In[22]:


#isdigit()? on 'horsepower'
hpIsDigit=pd.DataFrame(car_df.horsepower.str.isdigit())  #if the string is made of digit store True else 
#print isDigit=False!
car_df[hpIsDigit['horsepower']==False]


# In[23]:


car_df["horsepower"]=car_df["horsepower"].replace("?",np.nan)
car_df["horsepower"]=car_df["horsepower"].astype(float)


# In[24]:


median1=car_df["horsepower"].median()
median1


# In[25]:


car_df["horsepower"].replace(np.nan,median1,inplace=True)


# In[26]:


car_df[hpIsDigit['horsepower']==False]


# In[27]:


car_df.dtypes


# In[28]:


### Duplicates?
duplicate=car_df.duplicated() 
duplicate.sum()


# ## Bivariate Plots

# In[29]:


sns.pairplot(car_df,diag_kind="kde")


# ## split Data

# In[30]:


x= car_df.drop(['mpg'], axis=1)
y= car_df[['mpg']]


# In[31]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)


# ## Fit Linear Model

# In[33]:


model_1=LinearRegression()
model_1.fit(x_train, y_train)


# In[34]:


model_1.score(x_train,y_train)


# In[35]:


#out of sample score(R^2)

model_1.score(x_test,y_test)


# In[39]:


from sklearn.tree import DecisionTreeRegressor


# In[40]:


model_2=DecisionTreeRegressor()


# In[42]:


model_2.fit(x_train,y_train)


# In[45]:


model_2.score(x_train,y_train)


# In[46]:


model_2.score(x_test,y_test)


# In[ ]:




