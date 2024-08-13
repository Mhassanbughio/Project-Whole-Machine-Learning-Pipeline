#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.tree import DecisionTreeClassifier


# In[4]:


df=pd.read_csv("Titanic.csv")
df.head(3)


# ***Let's Plan***

# In[6]:


df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)


# In[9]:


X_train, X_test, y_train,y_test=train_test_split(df.drop(columns=['Survived']),
                                                 df['Survived'],test_size=0.2,random_state=42)


# In[10]:


X_train.head(2)


# In[11]:


y_train.head(2)


# In[ ]:





# In[12]:


# imputation transformer
trf1 = ColumnTransformer([
    ('impute_age',SimpleImputer(),[2]),
    ('impute_embarked',SimpleImputer(strategy='most_frequent'),[6])
],remainder='passthrough')


# In[14]:


# one hot encoding
trf2 = ColumnTransformer([
    ('ohe_sex_embarked',OneHotEncoder(sparse=False,handle_unknown='ignore'),[1,6])
],remainder='passthrough')


# In[15]:


# Scaling
trf3 = ColumnTransformer([
    ('scale',MinMaxScaler(),slice(0,10))
])


# In[16]:


# Feature selection
trf4 = SelectKBest(score_func=chi2,k=8)


# In[17]:


# train the model
trf5 = DecisionTreeClassifier()


# In[18]:


pipe = Pipeline([
    ('trf1',trf1),
    ('trf2',trf2),
    ('trf3',trf3),
    ('trf4',trf4),
    ('trf5',trf5)
])


# ## Pipeline Vs make_pipeline
# Pipeline requires naming of steps, make_pipeline does not.
# 
# (Same applies to ColumnTransformer vs make_column_transformer)

# In[ ]:


# Alternate Syntax
#pipe = make_pipeline(trf1,trf2,trf3,trf4,trf5)


# In[19]:


# train
pipe.fit(X_train,y_train)


# # Explore the Pipeline
# 

# In[21]:


# Code here
#pipe.named_steps


# In[22]:


# Display Pipeline
from sklearn import set_config
set_config(display='diagram')


# In[23]:


# Predict
y_pred = pipe.predict(X_test)


# In[24]:


y_pred


# In[25]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# ## Cross Validation using Pipeline
# 

# In[26]:


# cross validation using cross_val_score
from sklearn.model_selection import cross_val_score
cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean()


# ## GridSearch using Pipeline (Hyper_Prametertunnig )
# 

# In[28]:


# gridsearchcv
params = {
    'trf5__max_depth':[1,2,3,4,5,None]
}


# In[29]:


from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)


# In[30]:


grid.best_score_


# In[31]:


grid.best_params_


# ## Exporting the Pipeline

# In[32]:


# export 
import pickle
pickle.dump(pipe,open('pipe.pkl','wb'))


# In[33]:


pipe = pickle.load(open('pipe.pkl','rb'))


# In[34]:


test_input2 = np.array([2, 'male', 31.0, 0, 0, 10.5, 'S'],dtype=object).reshape(1,7)


# In[35]:


pipe.predict(test_input2)


# In[ ]:





# In[ ]:




