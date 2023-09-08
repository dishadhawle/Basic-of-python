#!/usr/bin/env python
# coding: utf-8

# In[61]:


### loanprediction.csv


# ### problem statement:Loan Approval Prediction Problem

# In[62]:


### step 1
### import the packages numpy,pandas,matplotlib,searborn,sklearn,train test split,metrics 


# In[63]:


### step2:Load the Dataset 


# In[64]:


### step3:Explore the data.shape,visualisation


# In[65]:


### step4:x,y-->train data test data--->fit the model with training data predict with the test data


# In[66]:


#Basic and most important libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#classifiers
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

#model evaluation tools
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

#Data processing functions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

import warnings
warnings.filterwarnings("ignore")


# In[67]:


data=pd.read_csv("loan_prediction.csv")
data.head(5)


# In[68]:


data.shape


# In[69]:


data.dtypes


# In[70]:


sns.countplot(x="Gender",hue="Loan_Status",data=data)


# In[71]:


sns.countplot(x="Married",hue="Loan_Status",data=data)


# In[72]:


correlation_mat=data.corr()


# In[73]:


sns.heatmap(correlation_mat,annot=True,linewidths=.5,cmap="YlGnBu")


# ### There is a positive correlation between Applicantincome and LoanAmount Coapplicantincome and LoanAmount

# In[74]:


sns.pairplot(data)
plt.show()


# In[75]:


data.describe()


# In[76]:


data.info()


# In[77]:


data.isnull().sum()


# In[78]:


plt.figure(figsize=(10,6))
sns.heatmap(data.isnull(),yticklabels=False)


# prepare data for model training i.e.removing outliers,filling null values

# In[79]:


print(data["Gender"].value_counts())
print(data["Married"].value_counts())
print(data["Self_Employed"].value_counts())
print(data["Dependents"].value_counts())
print(data["Credit_History"].value_counts())
print(data["Loan_Amount_Term"].value_counts())


# In[80]:


#Filling all non value with mode of respective variable
data["Gender"].fillna(data["Gender"].mode()[0],inplace=True)
data["Married"].fillna(data["Married"].mode()[0],inplace=True)
data["Self_Employed"].fillna(data["Self_Employed"].mode()[0],inplace=True)
data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].mode()[0],inplace=True)
data["Dependents"].fillna(data["Dependents"].mode()[0],inplace=True)
data["Credit_History"].fillna(data["Credit_History"].mode()[0],inplace=True)

#All value of "Dependent" columns were of "str" form now converting to "int" from.
data["Dependents"]=data["Dependents"].replace('3+',int(3))
data["Dependents"]=data["Dependents"].replace('1+',int(1))
data["Dependents"]=data["Dependents"].replace('2+',int(2))
data["Dependents"]=data["Dependents"].replace('0+',int(0))

data["LoanAmount"].fillna(data["LoanAmount"].median(),inplace=True)

print(data.isnull().sum())

#Heat map for null values
plt.figure(figsize=(10,6))
sns.heatmap(data.isnull())


# In[81]:


data.head(5)


# In[82]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[83]:


data["Gender"]=le.fit_transform(data["Gender"])
data["Married"]=le.fit_transform(data["Married"])
data["Education "]=le.fit_transform(data["Education"])
data["Self_Employed"]=le.fit_transform(data["Self_Employed"])
data["Property_Area"]=le.fit_transform(data["Property_Area"])
data["Loan_Status"]=le.fit_transform(data["Loan_Status"])

#data=pd.get_dummies(data)
data.head(5)


# In[84]:


#Dividing data into Input X variables and Target Y variable
x=data.drop(["Loan_Status","Loan_ID"],axis=1)
y=data["Loan_Status"]


# In[85]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[91]:


model=LogisticRegression(solver= "liblinear")


# In[92]:


model.fit(x_train,y_train)


# In[94]:


model.score(x_train,y_train)


# In[93]:


model.score(x_test,y_test)


# In[ ]:


dtree=DecisionTreeClassifier(criterion="gini")
dtree.fit(x_train,y_train)


# In[ ]:


dTreeR.score(x_train,y_train)


# In[ ]:


dTreeR.score(x_test,y_test)


# In[ ]:


dtree=DecisionTreeClassifier(criterion="gini",max_depth=3,random_state=0)
dtree.fit(x_train,y_train)
print(dTreeR.score(x_train,y_train))


# In[ ]:


y_predict=dTreeR.predict(x_test)


# In[ ]:


print(dTreeR.score(x_test,y_test))


# In[ ]:


from sklearn import metrics


# In[ ]:


cm=metrics.confusion_matrix(y_test,y_predict,labels=[0,1])

df_cm=pd.DataFrame(cm,index=[i for i in ["No","Yes"]],
                   columns=[i for i in ["No","Yes"]])
plt.figure(figsize=(7,5))
sns.heatmap(df_cm,annot=True,fmt='g')


# In[ ]:


from sklearn.ensemble import BaggingClassifier
bgcl=BaggingClassifier(n_estimators=150,base_estimator=dTreeR,random_state=0)
bgcl=bgcl.fit(x_train,y_train)
y_predict=bgcl.predict(x_test)
print(bgcl.score(x_test,y_test))


# In[ ]:


from skearn import metrics
cm=metrics.confusion_matrix(y_test,y_predict,labels=[0,1])

df_cm=pd.DataFrame(cm,index=[i for i in ["No","Yes"]],
                   columns=[i for i in ["No","Yes"]])
plt.figure(figsize=(7,5))
sns.heatmap(df_cm,annot=True,fmt='g')



# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
abcl=AdaBoostClassifier(n_estimators=120,base,random_state=0)
abcl=abcl.fit(x_train,y_train)
y_predict=abcl.predict(x_test)
print(abcl.score(x_test,y_test))


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gbcl=GrandientBoostingClassifier(n_estimators=200,random_state=0)
gbcl=gbcl.fit(x_train,y_train)
y_predict=gbcl.predict(x_test)
print(gbcl.score(x_test,y_test))


# In[ ]:


cm=metrics.confusion_matrix(y_test,y_predict,labels=[0,1])

df_cm=pd.DataFrame(cm,index=[i for i in ["No","Yes"]],
                   columns=[i for i in ["No","Yes"]])
plt.figure(figsize=(7,5))
sns.heatmap(df_cm,annot=True,fmt='g')


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfcl=RandomForestClassifier(n_estimators=160,random_state=0,max_features=3)
rfcl=rfcl.fit(x_train,y_train)


# In[ ]:


y_predict=rfcl.predict(x_test)
print(rfcl.score(x_test,y_test))
cm=metrics.confusion_matrix(y_test,y_predict,labels=[0,1])

df_cm=pd.DataFrame(cm,index=[i for i in ["No","Yes"]],
                   columns=[i for i in ["No","Yes"]])
plt.figure(figsize=(7,5))
sns.heatmap(df_cm,annot=True,fmt='g')


# In[ ]:




