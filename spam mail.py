#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data=pd.read_csv("spam.csv", encoding="latin-1")


# In[3]:


data.head()


# In[4]:


data.columns


# In[5]:


data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)


# In[6]:


data.head()


# In[7]:


data['class']=data['class'].map({'ham':0, 'spam':1})


# In[8]:


data.head()


# In[9]:


from sklearn.feature_extraction.text import CountVectorizer


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


X=data['message']
y=data['class']


# In[12]:


X.shape


# In[13]:


y.shape


# In[14]:


data.isnull().sum()


# In[15]:


cv=CountVectorizer()


# In[16]:


X=cv.fit_transform(X)


# In[17]:


x_train, x_test,y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)


# In[18]:


x_train.shape


# In[19]:


x_test.shape


# In[20]:


from sklearn.naive_bayes import MultinomialNB


# In[21]:


model=MultinomialNB()


# In[22]:


model.fit(x_train, y_train)


# In[23]:


model.score(x_test, y_test)


# In[24]:


msg="You Won 500$"
data = [msg]
vect = cv.transform(data).toarray()
my_prediction = model.predict(vect)


# In[25]:


vect


# In[26]:


import pickle
pickle.dump(model, open('spam.pkl','wb'))
model1 = pickle.load(open('spam.pkl','rb'))


# In[27]:


import pickle
pickle.dump(cv, open('vec.pkl','wb'))


# In[28]:




