#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


df=pd.read_csv("titanic-training-data.csv")


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# ## Univariate Analysis

# In[7]:


sns.countplot(x="Survived",data=df)


# In[8]:


sns.countplot(x="Pclass",data=df)
plt.show()


# In[9]:


sns.countplot(x="Embarked",data=df)
plt.show()


# In[10]:


import warnings
warnings.filterwarnings("ignore")


# In[11]:


sns.distplot(df["Age"])


# In[12]:


sns.boxplot(df["Age"])
plt


# In[13]:


sns.distplot(df["SibSp"])
plt.show()


# In[14]:


sns.distplot(df["Parch"])
plt.show()


# In[15]:


df.describe(include="all")


# ## Bivariate Analysis

# In[16]:


sns.countplot(x="Sex",hue="Pclass",data=df)


# In[17]:


sns.countplot(x="Embarked",hue="Pclass",data=df)


# In[18]:


sns.boxplot(x="Pclass",y="Age",data=df)


# In[19]:


sns.boxplot(x="Sex",y="Age",data=df)


# In[20]:


sns.boxplot(x="Embarked",y="Age",data=df)


# ## Multivariate Analysis 

# In[21]:


sns.violinplot(x="Sex",y="Age",data=df)


# In[22]:


sns.violinplot(x="Sex",y="Age",hue="Embarked",data=df)


# In[23]:


sns.violinplot(x="Pclass",y="Age",hue="Embarked",data=df)


# In[24]:


df=df.drop(columns=['PassengerId',"Ticket","Fare","Name"],axis=1)


# In[25]:


#sns.pairplot(df,hue="Survived")


# ### Missing Value Treatment 

# In[26]:


median1=df["Age"].median()
df["Age"]=df["Age"].fillna(median1)


# In[27]:


mode1=df["Embarked"].mode()[0]
df["Embarked"]=df["Embarked"].fillna(mode1)


# In[28]:


df.isnull().sum()


# ### Outlier Treatment 

# In[29]:


q1=df["Age"].quantile(0.25)
q3=df["Age"].quantile(0.75)
iqr=q3-q1


# In[30]:


lower_threshold=q1-1.5*iqr
upper_threshold=q3+1.5*iqr


# In[31]:


lower_threshold 


# In[32]:


upper_threshold


# In[33]:


df=df[(df["Age"]>lower_threshold)&(df["Age"]<upper_threshold )]


# In[34]:


sns.boxplot(df["Age"])
plt.show()


# In[35]:


sns.boxplot(df["Age"])


# ### Encoding

# In[38]:


df=pd.get_dummies(df,columns=["Sex","Embarked"])


# In[39]:


df.dtypes


# In[40]:


### Horsepower-->?????(median)


string cannot be converted to float->astype(float )


# In[ ]:




