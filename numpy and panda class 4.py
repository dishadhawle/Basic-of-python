#!/usr/bin/env python
# coding: utf-8

# 
# ### Numpy :-Package for Multidimension array
# 

# In[7]:


import numpy as np


# In[10]:


arr=np.array([1,2,3])
arr


# In[8]:


list_of_lists=[[1,2,3,],[4,5,6],[7,8,9]]
np.array(list_of_lists)


# In[9]:


simple_list=[6,7,9]
np.array(simple_list)


# In[11]:


np.arange(5,10)


# In[12]:


np.arange(1,100)


# In[13]:


np.arange(1,31,5)  # shift+tab->help box


# In[14]:


np.arange(5)


# In[15]:


np.zeros(10)


# In[17]:


np.zeros(10,int)


# In[18]:


np.ones((2,3))


# In[19]:


np.ones(100)


# In[20]:


np.ones(10,int)


# In[22]:


np.zeros((2,5),int)


# In[23]:


np.ones((2,5))


# In[24]:


np.ones((2,5),int)


# In[25]:


np.linspace(0,2,5)


# In[28]:


np.linspace(0,20,8)


# In[29]:


np.eye((10))


# In[30]:


np.random.rand(3,5)


# In[31]:


arr=np.random.randn(2,4)
arr


# In[32]:


np.random.randint(2,100)


# In[33]:


np.random.randint(20,56,100)


# In[34]:


sample_array=np.arange(30)
sample_array


# In[35]:


rand_array=np.random.randint(0,100,20)
rand_array


# In[40]:


sample_array.reshape(5,6)

sample_array.reshape(4,3)rand_array.max()
# In[41]:


rand_array.argmax()


# In[42]:


a=np.eye(5)
a


# In[43]:


a.T


# In[44]:


a=np.random.rand(2,3)
a


# In[45]:


a.T


# In[51]:


sample_array=np.arange(10,21)
sample_array


# In[47]:


sample_array[0]


# In[48]:


sample_array[2:5]


# In[49]:


sample_array[1:4]=100
sample_array


# In[50]:


sample_array=np.arange(10,21)
sample_array


# In[52]:


sample_array[0:7]


# In[53]:


subset_sample_array=sample_array[0:7]
subset_sample_array


# In[54]:


subset_sample_array[:]=1001
subset_sample_array


# In[ ]:





# 
# ### Two Dimensional array
# 

# In[56]:


import numpy as np


# In[58]:


sample_matrix=np.array([[50,20,1,23],[24,23,21,33],[56,76,24,7]])
sample_matrix


# In[59]:


sample_matrix[1,2]


# In[60]:


sample_matrix[2,:]


# In[61]:


sample_matrix[2]


# In[62]:


sample_matrix[:,(3,2)]


# In[ ]:




