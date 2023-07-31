#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd


# In[7]:


df=pd.read_csv("CardioGoodFitness-1.csv")


# In[8]:


df.head()


# In[9]:


df.tail()


# In[10]:


df.sample()


# In[11]:


df.sample(5)


# In[12]:


df.dtypes


# In[13]:


df.info()


# In[14]:


df.describe()


# In[15]:


df.describe(include="all")


# In[16]:


df.isnull().sum()


# In[ ]:




