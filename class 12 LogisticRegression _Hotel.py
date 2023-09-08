#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[37]:


df=pd.read_csv("hotel_bookings.csv")


# In[38]:


df.head()


# In[39]:


df.shape


# In[40]:


df.isnull().sum()


# In[41]:


df.dtypes


# In[42]:


median1=df["agent"].median()
median1


# In[43]:


df["agent"]=df["agent"].replace(np.nan,median1)


# In[44]:


median2=df["children"].mean()
median2


# In[45]:


df["children"]=df["children"].replace(np.nan,median2)


# In[46]:


model=df["country"].mode()[0]


# In[48]:


df["country"]=df["country"].replace(np.nan,median1)


# In[49]:


# df.hist(figsize=(10,20)
# plt.show()


# In[50]:


df.dtypes


# In[ ]:




