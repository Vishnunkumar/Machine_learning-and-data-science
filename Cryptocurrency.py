
# coding: utf-8


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts





data = pd.read_csv('Coin_data.csv')





data['Date'] = pd.to_datetime(data.Date, errors='coerce')





data.dtypes





data['Open'] = pd.to_numeric(data.Open, errors='coerce')
data['High'] = pd.to_numeric(data.High, errors='coerce')
data['Low'] = pd.to_numeric(data.Low, errors='coerce')
data['Close'] = pd.to_numeric(data.Close, errors='coerce')
data['Volume'] =pd.to_numeric(data.Volume, errors='coerce')
data['Market_Cap'] = pd.to_numeric(data.Market_Cap, errors='coerce')





data.dtypes





data.isnull().sum()





data.Date.fillna(method='ffill',inplace=True)
data.Open.fillna(method='ffill',inplace=True)
data.High.fillna(method='ffill',inplace=True)
data.Low.fillna(method='ffill',inplace=True)
data.Close.fillna(method='ffill',inplace=True)





train = data.iloc[:,1:7]
test = data.iloc[:,7:8]





print(train.head())
print(test.head())





from sklearn.preprocessing import Imputer as imp


# In[12]:


imp= imp(missing_values='NaN',strategy='median',axis=0)


# In[13]:


train.iloc[:,1:6] = imp.fit_transform(train.iloc[:,1:6])


# In[14]:


train.isnull().sum()


# In[15]:


test.fillna(pd.DataFrame.mean(test), inplace=True)


# In[16]:


test.isnull().sum()


# In[17]:


train.head()


# In[23]:


#Replacing Date with some usable measure
train['Date'] = (train.Date.dt.month)*10**(-2) + train.Date.dt.dayofyear


# In[25]:


from sklearn.ensemble import BaggingRegressor as BR


# In[28]:


X_tr, X_te, Y_tr, Y_te = tts(train, test, test_size=.3, random_state=255)


# In[29]:


br.fit(X_tr,Y_tr)


# In[30]:


pred = br.predict(X_te)


# In[34]:


pred.shape


# In[37]:


Y_te = np.resize(Y_te,(24947,))


# In[39]:


Pred = Y_te - pred


# In[41]:


br.score(X_te,Y_te)


# In[42]:


Pred = np.sum((Pred)**2)


# In[43]:


Pred


# In[44]:


Avg = Y_te - np.mean(Y_te)


# In[45]:


Avg = np.sum((Avg)**2)


# In[46]:


Avg


# In[48]:


R = Pred/Avg


# In[49]:


1-R

