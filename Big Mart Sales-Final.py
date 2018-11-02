
# coding: utf-8

# In[24]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
from sklearn.cross_validation import train_test_split as tts
import pandas as pd
from sklearn.preprocessing import Imputer


# In[25]:


from collections import Counter
#used to find frequency in the list


# In[26]:


train = pd.read_csv('TrainB.csv')
test = pd.read_csv('TestB.csv')
test.isnull().sum()
train.isnull().sum()


# # DATA PREPROCESSING 

# In[31]:


median_value_tr= train['Outlet_Size'].median()
train['Outlet_Size'] = train['Outlet_Size'].fillna(median_value_tr)
median_value1_tr= train['Item_Weight'].median()
train['Item_Weight'] = train['Item_Weight'].fillna(median_value1_tr)


# In[32]:


median_value2_tr= test['Item_Weight'].median()
test['Item_Weight'] = test['Item_Weight'].fillna(median_value2_tr)
median_value_te= test['Outlet_Size'].median()
test['Outlet_Size'] = test['Outlet_Size'].fillna(median_value_te)


# In[33]:


print(train.shape)
print(test.shape)


# In[35]:


train.isnull().sum()
test.isnull().sum()


# In[38]:


tr_x = train.iloc[0:8523,1:10]
te_x = test.iloc[0:5681,1:10]
tr_y = train.iloc[0:8523,10]
print(tr_y.head())
Counter(tr_x.Outlet_Type)
print(tr_x.head())


# In[39]:


#function to replace missing values
def transcore(x):
    if x == "Baking Goods":
        return 0
    if x == "Breads":
        return 1
    if x == "Break fast":
        return 2
    if x == "Canned":
        return 3
    if x == "Dairy":
        return 4
    if x == "Frozen Foods":
        return 5
    if x == "Fruits and Vegetables":
        return 6
    if x == "Hard Drinks":
        return 7
    if x == "Health and Hygiene":
        return 8
    if x == "Household":
        return 9
    if x == "Meat":
        return 10
    if x == "Others":
        return 11
    if x == "Seafood":
        return 12
    if x == "Snack Foods":
        return 13
    if x == "Soft Drinks":
        return 14
    if x == "Starchy Foods":
        return 15
    if x == " ":
        return 16
    


# In[40]:


tr_x.isnull().sum()


# In[41]:


def groupscore(x):
    if x == "Grocery Store":
        return 0
    if x == "Supermarket Type1":
        return 1
    if x == "Supermarket Type2":
        return 2
    if x == "Supermarket Type3":
        return 3


# In[43]:


tr_x['New_Type'] = tr_x['Item_Type'].apply(transcore)
#axis=1 indicates column while axis=0 indicates row
tr_x.drop('Item_Type', axis=1,inplace=True)
te_x['New_Type'] = te_x['Item_Type'].apply(transcore)
te_x.drop('Item_Type', axis=1,inplace=True)
#inplace=True defines that drop operation should take place
#inplace=True defines that drop operation should take place


# In[44]:


#ensuring that all data are pre-processed for mathematical applications
print(tr_x.dtypes)
print(te_x.dtypes)


# In[45]:


#checking missing values in a dataframe
print(te_x.isnull().sum())
print(tr_x.isnull().sum())


# In[46]:


tr_x['Market_Type'] = tr_x['Outlet_Type'].apply(groupscore)
#axis=1 indicates column while axis=0 indicates row
tr_x.drop('Outlet_Type', axis=1,inplace=True)
te_x['Market_Type'] = te_x['Outlet_Type'].apply(groupscore)
te_x.drop('Outlet_Type', axis=1,inplace=True)


# In[47]:


print(tr_x.isnull().sum())
print(te_x.isnull().sum())


# # USING MEDIAN VALUES TO FILLNA IN NEW_TYPE

# In[48]:


tr_x['New_Type']=tr_x['New_Type'].fillna(tr_x['New_Type'].median())
te_x['New_Type']=te_x['New_Type'].fillna(te_x['New_Type'].median())


# In[49]:


tr_x.isnull().sum()


# In[50]:


tr_x.shape


# In[51]:


tr_y.head()


# In[52]:


tr_y.shape


# # Checking Dimensionality and missing values

# In[53]:


tr_x.isnull().sum()


# In[55]:


te_x.isnull().sum()


# In[56]:


reg = lm.LinearRegression()


# # MATHEMATICAL MODELLING

# In[57]:


reg.fit(tr_x,tr_y)


# In[58]:


reg.coef_


# In[59]:


pred = reg.predict(te_x)


# In[60]:


Item_Outlet_Sales = pred


# In[61]:


#Sales is predicted and given as output in Outlet_sales
Item_Outlet_Sales


# # EXPORTING CSV

# In[65]:


Submit = pd.read_csv("SS.csv")


# In[68]:


print(Submit.columns)
Submit.head()
Submit.shape


# In[69]:


Final = Submit.iloc[0:5681,0:2]
Final.head(2)


# In[70]:


Sales = pd.DataFrame({'Item_Identifier':Final.Item_Identifier.values,'Item_Outlet_Sales':Item_Outlet_Sales,'Outlet_Identifier':Final.Outlet_Identifier.values})
Sales


# In[71]:


Sales.to_csv("Martsales.csv", encoding='utf-8')


# # End of Program
