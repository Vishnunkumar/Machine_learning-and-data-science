import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

r_d = os.path.abspath(r'E:\Placements\Datasets\black-friday')
data = pd.read_csv(os.path.join(r_d,'data.csv'))

#Checking possible unneccessary 
print(data.isnull().sum())
print(data.dtypes)
print(data.shape)

Product_ID = data['Product_ID']
data.drop(['Product_ID'],axis=1,inplace=True)

#conversion of string to int
objects = data.select_dtypes(['object']).columns
data[objects] = data[objects].apply(lambda x:x.astype('category'))
cat = data.select_dtypes(['category']).columns
data[cat] = data[cat].apply(lambda x:x.cat.codes)

#
data.columns = ['User_ID','Gender','Age','Occupation','City','Stay','Marital','PC1','PC2','PC3','Purchase']
data.drop(['PC3'],axis=1,inplace=True)
data.fillna(method='ffill',inplace=True)

#Removal of ouliers
m = data.mean()
s = data.std()
data_out = data[(data > m-2*s) & (data < m +2*s)]
data_out.dropna(how='any',axis=0,inplace=True)

data = data_out
Result = data['Purchase']
Features = data.iloc[:,1:9]
Result_tr = Result.loc[0:327154]
Features_tr = Features.loc[0:327153,:]
tr_x,te_x,tr_y,te_y = tts(Features_tr,Result_tr,test_size=.2,random_state=255)
data.drop(['User_ID'],axis=1,inplace=True)
GBR = GBR(max_features=4,n_estimators=50,loss='ls')
GBR.fit(tr_x,tr_y)

