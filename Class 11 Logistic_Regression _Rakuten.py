#!/usr/bin/env python
# coding: utf-8

# In[131]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[132]:


df=pd.read_csv("titanic-training-data.csv")


# In[133]:


df.shape


# In[134]:


df.head()


# In[135]:


df.info()


# In[136]:


df.isnull().sum()


# ### EDA

# In[137]:


### Analyze ddependent variable
sns.countplot(x="Survived",data=df)


# In[138]:


sns.countplot(x="Survived",hue="Pclass",data=df)


# In[139]:


df.drop(["PassengerId","Name","Ticket","Fare"],axis=1,inplace=True)
df.head()


# In[140]:


df.hist(figsize=(20,30))
plt.show


# In[141]:


sns.countplot(x="SibSp",data=df)


# In[142]:


sns.heatmap(df.isnull(),yticklabels=False,cmap="viridis")


# In[143]:


sns.boxplot(x="Pclass",y="Age",data=df)


# In[144]:


df.dropna(inplace=True)


# In[145]:


sns.heatmap(df.isnull(),yticklabels=False,cmap="viridis")


# In[146]:


df.isnull().sum()


# In[147]:


pd.get_dummies(df,columns=["Pclass","Sex","Embarked"])


# In[148]:


df.isnull().sum()


# In[149]:


df.dtypes


# In[150]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[151]:


x=df.drop(["Survived"],axis=1)

y=df[['Survived']]


# In[152]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)


# ### Fit the model

# In[153]:


pd.get_dummies(df,columns=["Pclass","Sex","Embarked"])


# In[154]:


df.dtypes


# In[155]:


import warnings 
warnings.filterwarnings("ignore")


# In[156]:


model=LogisticRegression(solver="lbfgs")
model.fit(x_train, y_train)
model


# In[ ]:


model.fit(x_train, y_train)


# In[ ]:





# In[157]:


model.score(x_train,y_train)


# In[ ]:


model.score(x_test,y_test)


# In[ ]:


predictions=model.predict(x_test)


# In[ ]:


from sklearn import metrics


# In[ ]:


print(metrics.classification_report(y_test,predictions))


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[ ]:


cm=metrics.confusion_matrix(y_test,predictions,label=[1,0])

df_cm=pd.DataFrame(cm, index=[i for i in["1","0"]],
                   columns =[i for i in["predict1","predict0"]])
plt.figure(figsize =(7,5))
sns.heatmap(df_cm,annot=True,fmt='g')


# In[ ]:





# In[ ]:




