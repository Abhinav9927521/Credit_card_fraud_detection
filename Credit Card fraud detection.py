#!/usr/bin/env python
# coding: utf-8

# # CREDT CARD FRAUD DETECTION

# In[30]:


# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# In[31]:


# Load the dataset from the csv file using pandas
df= pd.read_csv("fraudTest.csv")


# In[32]:


df


# In[33]:


# Display first five row from the dataset
df.head()


# In[34]:


# Display last five rows from the dataset
df.tail()


# In[35]:


# Print the shape of the data
df.shape


# In[36]:


# Display description of the dataset
df.describe()


# In[37]:


# Checking any missing value in the dataset
df.isnull()


# In[38]:


df.isnull().sum()


# In[39]:


df.info()


# In[40]:


df.drop(['Unnamed: 0'],axis=1,inplace=True)


# In[41]:


df.isnull().sum()


# In[42]:


# Count the number of values in the gender column
df.gender.value_counts()


# In[43]:


df['male']=df.gender.apply(lambda x: 1 if x=="M" else 0)


# In[44]:


df.drop(['gender','category','cc_num','merchant','trans_date_trans_time','first','last','street','state','city','city_pop','job','dob','unix_time','trans_num'],axis=1,inplace=True)
df.head()


# In[45]:


df.dtypes


# In[46]:


df.head()


# In[47]:


# Correlation matrix
corr_mat = df.corr()
fig = plt.figure(figsize = (10, 4))
sns.heatmap(corr_mat, vmax = .8, square = True)
plt.show()


# In[48]:


# Select the features and the target variable.
x=df.drop(['is_fraud'],axis=1)
y=df['is_fraud']


# In[49]:


print(x.shape)
print(y.shape)


# In[50]:


# Split the dataset into training and testing sets using train_test_split from sklearn.model_selection. 
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=2)


# In[51]:


# Import the LinearRegression class from sklearn.linear_model.
lr_model=LogisticRegression()
lr_model.fit(xtrain,ytrain)


# In[52]:


lr_model.score(xtest,ytest)


# In[53]:


lr_pred=lr_model.predict(xtest)


# In[54]:


lr_pred


# In[55]:


# Display the model's coefficients and intercept.
print('cofficient:',lr_model.coef_)
print('intercept:', lr_model.intercept_)

