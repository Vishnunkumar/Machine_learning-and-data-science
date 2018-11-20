import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

r_d = pd.read_csv(os.path.abspath(r'D:\Datas\seattle.csv'))
data.fillna(method='ffill',inplace=True)
data['RAIN'] = [1 if x == True else 0 for x in data['RAIN']]
data['DATE'] = pd.to_datetime(data['DATE'])
data.columns = ['Date','Prcp','Tmax','Tmin','Rain']
data['Tdiff'] = data['Tmax'] - data['Tmin']
data['Year'] = data['Date'].dt.year
data['Day'] = (data['Date'].dt.month)*(.01) + data['Date'].dt.dayofyear
data_n = data
drops = ['Date','Tmax','Tmin']
data.drop(drops,inplace=True,axis=1)
Rain = data['Rain']
data.drop(['Rain'],inplace=True,axis=1)
from sklearn.linear_model import LogisticRegression 

from sklearn.cross_validation import train_test_split as tts
x_tr,x_te,y_tr,y_te = tts(data,Rain,train_size=.7,random_state=255)

#Rain prediction LR
Accuracy=[]
for i in range(1,17):
    from sklearn.linear_model import LogisticRegression 
    x_tr,x_te,y_tr,y_te = tts(data,Rain,train_size=.7,random_state=255)
    LR = LogisticRegression()
    LR.fit(x_tr,y_tr)
    Accuracy.append(LR.score(x_te,y_te))

#Rain prediction RFC

accuracy = []
for i in range(1,17):
    from sklearn.ensemble import RandomForestClassifier as RFC
    RFC = RFC(n_jobs=4,n_estimators=i,max_leaf_nodes=3)
    RFC.fit(x_tr,y_tr)
    accuracy.append(RFC.score(x_te,y_te))
    
#plot of LR_accuracy vs RFC_accuracy
fig = plt.figure(figsize=(8,8))
plt.plot(np.arange(1,17),accuracy,c='g')
plt.plot(np.arange(1,17),Accuracy,c='r')
plt.xlabel('No_iters')
plt.ylabel('accuracy')
plt.title('RFC_Accuracy vs LR_Accuracy')

    