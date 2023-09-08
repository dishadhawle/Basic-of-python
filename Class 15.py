#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings 
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)


# In[2]:


plt.style.use('dark_background')


# In[ ]:


df=pd.read_csv("pred")

