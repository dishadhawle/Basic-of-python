#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


##load the dataset
df=pd.read_csv("CardioGoodFitness-1.csv")


# In[3]:


df.head()


# In[4]:


df.shape


#         There are 180 rows and 9 columns

# In[5]:


df.dtypes


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df.describe(include="all")


# In[9]:


df.hist(figsize=(10,10))
plt.show()


# In[14]:


sns.countplot(x='Product',data=df)


# In[15]:


sns.countplot(x='Gender',data=df)


# In[17]:


sns.countplot(x='MaritalStatus',data=df)


# In[18]:


sns.countplot(x='Product',hue="Gender",data=df)


# In[19]:


sns.boxplot(x="Age",data=df)


# In[20]:


iqr=33-24


# In[21]:


iqr


# In[22]:


24-1.5*9


# In[23]:


33+1.5*9


# In[24]:


sns.boxplot(df)


# In[28]:


sns.boxplot(x="Age",data=df,palette="Set2")


# In[31]:


sns.boxplot(x="Product",y="Age",data=df,palette="Set2")


# In[33]:


sns.boxplot(x="Product",y="Income",data=df)


# In[34]:


sns.pairplot(df)


# In[35]:


corr=df.corr()
corr


# In[36]:


sns.heatmap(corr,annot=True)


# In[ ]:




