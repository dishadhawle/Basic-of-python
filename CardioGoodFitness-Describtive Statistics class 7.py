#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


mydata=pd.read_csv("CardioGoodFitness-1.csv")


# In[5]:


##top 5 rows
mydata.sample(10)


# In[6]:


mydata.shape


# In[7]:


mydata.dtypes


# In[8]:


mydata.columns


# In[9]:


mydata.info()


# In[10]:


### There are 180 non null observation in all the attributes which indicates that are no missing value 


# In[11]:


mydata.isnull().sum()


# In[12]:


mydata.describe()


# In[13]:


sns.boxplot(x="Age",data=mydata)


# In[15]:


import warnings
warnings.filterwarnings("ignore")


# In[16]:


sns.distplot(mydata["Age"])
plt.show()


# In[17]:


mydata.hist(figsize=(10,20))
plt.show()


# In[22]:


mydata.describe(include="all").T


# In[25]:


sns.countplot(x='MaritalStatus',data=mydata)


# In[26]:


sns.countplot(x="Product",hue="Gender",data=mydata)


# In[29]:


sns.countplot(x="Product",hue="MaritalStatus",data=mydata)


# In[31]:


sns.boxplot(x="Product",y="Age",data=mydata) ## Boxplot-one should be numerical


# In[32]:


pd.crosstab(mydata['Product'],mydata['Gender'])


# In[33]:


sns.pairplot(mydata,diag_kind='kde')


# In[34]:


corr=mydata.corr()
corr


# In[43]:


mydata["Age"].std()


# In[44]:


mydata["Age"].mean()


# In[ ]:




