import numpy as np
import os
import pandas as pd

r_d = os.path.abspath(r'E:\Placements\Datasets\Beer')
data = pd.read_csv(os.path.join(r_d,'recipeData.csv'),encoding='latin-1')

#checking anamalies
print(data.isnull().sum())
print(data.dtypes)
print(data.shape)

#check for categorical variables
print(data['SugarScale'].value_counts(),data['BrewMethod'].value_counts(),data['PrimingMethod'].value_counts(),data['PrimingAmount'].value_counts())

#Conversion to category variables
data['SugarScale'] = data['SugarScale'].astype('category')
data['BrewMethod'] = data['BrewMethod'].astype('category')

c_c = data.select_dtypes(['category']).columns
data[c_c] = data[c_c].apply(lambda x:x.cat.codes)

data.drop(['PrimingMethod'],axis=1,inplace=True)
data.drop(['PrimingAmount'],axis=1,inplace=True)

data_k = data

#dropping insignificant variables
data_k.drop(['Style'],axis=1,inplace=True)
data_k.drop(['URL'],axis=1,inplace=True)
data_k.drop(['Name'],axis=1,inplace=True)
data_k.drop(['StyleID'],axis=1,inplace=True)
data_k.drop(['BeerID'],axis=1,inplace=True)

#filling missing values using imputation
Imp = Imp(missing_values='NaN',strategy='mean',axis=1)
data_r = Imp.fit_transform(data_k)
data_r = pd.DataFrame(data_r)
data_r.columns = data_k.columns

import matplotlib.pyplot as plt

m = data_r.mean()
s = data_r.std()

data = data_r[(data_r >= m-2*s) & (data_r <= m+2*s)]
data.dropna(how='any',axis=0,inplace=True)

from sklearn.cross_validation import train_test_split as tts
from sklearn.feature_selection import RFE

data['Efficiency'] = data['Efficiency'].astype('int')
Class = data['BrewMethod']
data.drop(['BrewMethod'],inplace=True,axis=1)

tr_x,te_x,tr_y,te_y = tts(data,Class,train_size=.7,random_state=255)
from sklearn.ensemble import RandomForestClassifier as RFC
RFC = RFC(max_features=8,n_estimators=7,n_jobs=20)
RFC.fit(tr_x,tr_y)
RFC.score(te_x,te_y)

#wrt n_jobs
Accuracy = []
for i in range(1,20):    
    from sklearn.ensemble import RandomForestClassifier as RFC
    RFC = RFC(max_features=8,n_estimators=7,n_jobs=i)
    RFC.fit(tr_x,tr_y)
    Accuracy.append(RFC.score(te_x,te_y))
accuracy = np.array(Accuracy)
plt.plot(np.arange(1,20),accuracy)
plt.xlabel('iterations')
plt.ylabel('accuracy')

#wrt n_estimators
Accuracy = []
for i in range(1,7):    
    from sklearn.ensemble import RandomForestClassifier as RFC
    RFC = RFC(max_features=8,n_estimators=i,n_jobs=10)
    RFC.fit(tr_x,tr_y)
    Accuracy.append(RFC.score(te_x,te_y))
accuracy = np.array(Accuracy)
plt.plot(np.arange(1,7),accuracy)
plt.xlabel('iterations')
plt.ylabel('accuracy')

#wrt max_features
Accuracy = []
for i in range(1,8):    
    from sklearn.ensemble import RandomForestClassifier as RFC
    RFC = RFC(max_features=i,n_estimators=7,n_jobs=10)
    RFC.fit(tr_x,tr_y)
    Accuracy.append(RFC.score(te_x,te_y))
accuracy = np.array(Accuracy)
fig = plt.figure(figsize=(8,8))
plt.plot(np.arange(1,8),accuracy)
plt.xlabel('iterations')
plt.ylabel('accuracy')    















