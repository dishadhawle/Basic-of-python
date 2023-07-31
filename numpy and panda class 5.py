#!/usr/bin/env python
# coding: utf-8

# 
# ### Selection techniques
# 
# 

# In[17]:


sample_array=np.arange(1,20)


# In[16]:


sample_array


# In[18]:


sample_array+sample_array


# In[19]:


np.exp(sample_array)


# In[20]:


np.sqrt(sample_array)


# In[21]:


np.max(sample_array)


# In[22]:


np.min(sample_array)


# In[23]:


np.argmax(sample_array)


# In[24]:


np.argmin(sample_array)


# In[25]:


np.square(sample_array)


# In[26]:


np.std(sample_array)


# In[27]:


np.var(sample_array)


# In[28]:


np.mean(sample_array)


# In[29]:


array=np.random.randn(3,4)
array


# In[12]:


np.round(array,decimals=2)


# In[13]:


sports=np.array(['golf','cricket','fball','cricket'])
np.unique(sports)


# 
# 
# ### Pandas
# 
# 

# In[15]:


import pandas as pd
import numpy as np


# In[3]:





# In[16]:


### pandas dataframe and indexing


# In[19]:


sports1=pd.Series([1,2,3,4],index=['cricket','football','basketball','golf'])
sports1


# In[20]:


sports1['football']


# In[33]:


sports2=pd.Series([11,2,3,4],index=['cricket','football','baseball','golf'])
sports2


# In[34]:


sports1+sports2


# In[ ]:


import pandas as pd
import numpy as np


# In[36]:


df1=pd.DataFrame(np.random.rand(8,5),index='A B C D E F G H'.split(),columns='score1 score2 score3 score4 score5'.split())
df1


# In[37]:


df1["score1"]


# In[38]:


df1[["score1","score2","score3"]]


# In[39]:


df1['score6']=df1['score1']+df1['score2']
df1


# In[43]:


df2={'ID':['101','102','103','107','176'],'Name':['John','Mercy','Akash','Kavin','Lally'],'profit':[20,54,56,87,123]}
df=pd.DataFrame(df2)
df


# In[60]:


df


# In[69]:


df.drop(3)


# In[ ]:




